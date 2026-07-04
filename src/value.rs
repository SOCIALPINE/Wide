//! Runtime values.
//!
//! Collections (vector/map/heap) are `Rc<RefCell<…>>` — reference semantics + mutable.
//! Needed so the push/pop/sort that algorithms require can work in place.
//! (Canon §4.3 governs this with an ownership model, but this is the pragmatic choice at the interpreter stage.)

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

pub type ArrayRef = Rc<RefCell<Vec<Value>>>;
pub type MapRef = Rc<RefCell<BTreeMap<MapKey, Value>>>;
pub type HeapRef = Rc<RefCell<Vec<Value>>>; // backing vector kept in min-heap order
pub type StructRef = Rc<RefCell<BTreeMap<String, Value>>>;
pub type SetRef = Rc<RefCell<BTreeSet<MapKey>>>; // std/set — deterministic element order
pub type StrBufRef = Rc<RefCell<String>>; // mutable string builder — amortized O(1) append
#[cfg(feature = "ai")]
pub type TensorRef = Rc<RefCell<TensorData>>;

/// Residence — §4.3. Computation is (still) on the CPU, but the *transfer cost model* is genuinely tracked and illuminated.
#[cfg(feature = "ai")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device {
    Host,
    Gpu,
}

#[cfg(feature = "ai")]
impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Device::Host => write!(f, "host"),
            Device::Gpu => write!(f, "gpu"),
        }
    }
}

/// Autodiff tape node — *how* the result tensor was produced (for backprop VJP). (Anatomy 7, A1·A4.)
#[cfg(feature = "ai")]
#[derive(Debug)]
pub enum GradOp {
    Add,            // dA=g, dB=g
    Sub,            // dA=g, dB=-g
    MulElem,        // dA=g*B, dB=g*A
    ScalarMul(f32), // dA=g*s (tensor-scalar +,-,*,/ all reduce to this form)
    MatMul,         // dA=g@Bᵀ, dB=Aᵀ@g
    Sum,            // dInput = g[0] broadcast
    Mean,           // dInput = g[0]/N
    SumAxis { axis: usize, rows: usize, cols: usize },  // 2D reduce along axis; dInput broadcasts g back
    MeanAxis { axis: usize, rows: usize, cols: usize }, // same, divided by the reduced count
    Reshape,        // view with a new shape (same data order); dInput = g reshaped to the input shape
    Conv2d,         // valid 2D cross-correlation; dX = full-corr of g with flipped k, dK = valid-corr of x with g
    MaxPool2d { k: usize }, // non-overlapping k×k max pooling; dInput routes g to each window's argmax
    Relu,           // dInput = g * (input>0)
    Transpose,      // dInput = gᵀ
    Sigmoid,        // s=σ(x); dInput = g * s*(1-s)
    Tanh,           // t=tanh(x); dInput = g * (1 - t²)
    Exp,            // e=eˣ; dInput = g * e
    Log,            // dInput = g / x
    Softmax,        // row-wise softmax; dInput[i] = s[i]·(g[i] - Σⱼ gⱼsⱼ)
}

#[cfg(feature = "ai")]
#[derive(Debug)]
pub struct GradNode {
    pub op: GradOp,
    pub inputs: Vec<TensorRef>,
}

/// Tensor — f32 flat buffer + shape + residence + (optional) grad tracking. The soul of wide (AI). Reference semantics.
/// Even on a slow interpreter, each op is a single Rust call so it's fast (the Python+NumPy lesson, T6).
#[cfg(feature = "ai")]
#[derive(Clone, Debug)]
pub struct TensorData {
    pub dtype: &'static str, // "f32"
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub device: Device,
    pub requires_grad: bool,
    pub grad: Option<Vec<f32>>,
    pub grad_fn: Option<Rc<GradNode>>, // tape link (None for leaves)
    // Adam optimizer state (per-parameter moment estimates + timestep) — lazily initialized by adam_step.
    pub adam_m: Option<Vec<f32>>,
    pub adam_v: Option<Vec<f32>>,
    pub adam_t: u32,
    // Real GPU residency (`gpu` feature, v0.50): the uploaded wgpu buffer, cached so chained gpu ops
    // re-upload nothing (§4.3). Must be invalidated whenever `data` is mutated in place.
    #[cfg(feature = "gpu")]
    pub gpu_buf: Option<std::rc::Rc<wgpu::Buffer>>,
}

// grad_fn is a graph (not a cycle) so it's excluded from comparison — value equality is dtype/shape/data/residence only.
#[cfg(feature = "ai")]
impl PartialEq for TensorData {
    fn eq(&self, o: &Self) -> bool {
        self.dtype == o.dtype && self.shape == o.shape && self.data == o.data && self.device == o.device
    }
}

#[cfg(feature = "ai")]
impl TensorData {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        TensorData {
            dtype: "f32",
            shape,
            data,
            device: Device::Host,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            adam_m: None,
            adam_v: None,
            adam_t: 0,
            #[cfg(feature = "gpu")]
            gpu_buf: None,
        }
    }
    pub fn size(&self) -> usize {
        self.shape.iter().product::<usize>().max(if self.shape.is_empty() { 1 } else { 0 })
    }
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }
}

/// A function value (v0.42 closures). `captured` holds the free variables *by value at creation* —
/// scalars are copied; collections/structs/tensors are `Rc` so they stay shared (the language's normal
/// reference semantics). Mutating a captured *scalar* inside the closure is per-call local (honest limit).
#[derive(Debug)]
pub struct FnData {
    pub name: Option<String>, // Some for a named fn used as a value; None for a lambda
    pub params: Vec<String>,
    pub body: Vec<crate::ast::Stmt>,
    pub captured: std::collections::HashMap<String, Value>,
}

// Function values compare by identity (same closure object), like Python.
impl PartialEq for FnData {
    fn eq(&self, o: &Self) -> bool {
        std::ptr::eq(self, o)
    }
}

/// Pointer provenance (memory model, §3.1/§3.4). Models `&x[i]`: which buffer it came from (`origin`),
/// where in it (`index`), and the source name for illumination. The interpreter has no real addresses,
/// so a pointer is a tracked reference into an Rc-backed buffer — the *illumination* is real (§3.7).
#[derive(Clone, Debug, PartialEq)]
pub struct PtrData {
    pub origin: ArrayRef,
    pub index: usize,
    pub origin_name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(ArrayRef),
    Map(MapRef),
    Heap(HeapRef),
    Set(SetRef),
    StrBuf(StrBufRef), // mutable string builder (reference semantics) — avoids `s = s + c` O(n²)
    Struct { name: String, fields: StructRef }, // mutable, reference semantics
    Enum { name: String, variant: String, payload: Vec<Value> },
    Err(Box<Value>), // error-value (Zig-style error union: "value or error")
    Fn(Rc<FnData>), // function value / closure (v0.42) — first-class, identity equality
    Ptr(std::rc::Rc<PtrData>), // pointer into a buffer (memory model, §3.1)
    #[cfg(feature = "ai")]
    Tensor(TensorRef), // tensor (AI) — f32, reference semantics
    Range(i64, i64), // half-open [start, end)
    Unit,
}

/// Map key — only the hashable/orderable subset (int·str·bool). Float keys are forbidden (bad practice).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapKey {
    Int(i64),
    Str(String),
    Bool(bool),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Float(_) => "f32",
            Value::Bool(_) => "bool",
            Value::Str(_) => "str",
            Value::Array(_) => "array",
            Value::Map(_) => "map",
            Value::Heap(_) => "heap",
            Value::Set(_) => "set",
            Value::StrBuf(_) => "strbuf",
            Value::Struct { .. } => "struct",
            Value::Enum { .. } => "enum",
            Value::Err(_) => "error",
            Value::Fn(_) => "fn",
            Value::Ptr(_) => "ptr",
            #[cfg(feature = "ai")]
            Value::Tensor(_) => "tensor",
            Value::Range(..) => "range",
            Value::Unit => "()",
        }
    }

    pub fn truthy(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn array(items: Vec<Value>) -> Value {
        Value::Array(Rc::new(RefCell::new(items)))
    }

    pub fn empty_map() -> Value {
        Value::Map(Rc::new(RefCell::new(BTreeMap::new())))
    }

    pub fn empty_heap() -> Value {
        Value::Heap(Rc::new(RefCell::new(Vec::new())))
    }

    /// Convert to a map key (only int·str·bool allowed).
    pub fn as_key(&self) -> Option<MapKey> {
        match self {
            Value::Int(n) => Some(MapKey::Int(*n)),
            Value::Str(s) => Some(MapKey::Str(s.clone())),
            Value::Bool(b) => Some(MapKey::Bool(*b)),
            _ => None,
        }
    }
}

impl MapKey {
    pub fn to_value(&self) -> Value {
        match self {
            MapKey::Int(n) => Value::Int(*n),
            MapKey::Str(s) => Value::Str(s.clone()),
            MapKey::Bool(b) => Value::Bool(*b),
        }
    }
}

/// Order-comparison of values (shared by sort/heap/comparison operators). Number-to-number / string-to-string only.
pub fn value_cmp(a: &Value, b: &Value) -> Result<Ordering, String> {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => Ok(x.cmp(y)),
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .ok_or_else(|| "not comparable (NaN)".to_string()),
            _ => Err(format!("cannot order-compare {} and {}", a.type_name(), b.type_name())),
        },
    }
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Float(x) => Some(*x),
        _ => None,
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(x) => write!(f, "{}", x),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Str(s) => write!(f, "{}", s),
            Value::Array(xs) => {
                write!(f, "[")?;
                for (i, v) in xs.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k.to_value(), v)?;
                }
                write!(f, "}}")
            }
            Value::Heap(h) => write!(f, "heap({})", h.borrow().len()),
            Value::Set(s) => {
                write!(f, "set{{")?;
                for (i, k) in s.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", k.to_value())?;
                }
                write!(f, "}}")
            }
            Value::StrBuf(s) => write!(f, "{}", s.borrow()),
            Value::Struct { name, fields } => {
                write!(f, "{} {{", name)?;
                for (i, (k, v)) in fields.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " {}: {}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::Enum { name, variant, payload } => {
                write!(f, "{}::{}", name, variant)?;
                if !payload.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in payload.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Err(p) => write!(f, "Err({})", p),
            Value::Fn(c) => match &c.name {
                Some(n) => write!(f, "fn {}({})", n, c.params.join(", ")),
                None => write!(f, "fn({})", c.params.join(", ")),
            },
            Value::Ptr(p) => write!(f, "ptr→{}[{}]", p.origin_name, p.index),
            #[cfg(feature = "ai")]
            Value::Tensor(t) => {
                let t = t.borrow();
                // a scalar tensor (1 element) prints as a bare number — so loss/reduction results look clean.
                if t.data.len() == 1 {
                    return write!(f, "{}", t.data[0]);
                }
                let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
                let dev = if t.device == Device::Gpu { "@gpu" } else { "" };
                write!(f, "tensor<{}>{}[{}]", t.dtype, dev, shape.join(", "))?;
                if t.data.len() <= 12 {
                    let vals: Vec<String> = t.data.iter().map(|x| format!("{}", x)).collect();
                    write!(f, " {{{}}}", vals.join(", "))?;
                }
                Ok(())
            }
            Value::Range(a, b) => write!(f, "{}..{}", a, b),
            Value::Unit => write!(f, "()"),
        }
    }
}
