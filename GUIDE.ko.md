# wide 프로그래밍 가이드

wide는 비용과 위험을 숨기거나 금지하는 대신 **보여주는** 언어입니다. 프로그램을 실행하면 결과와
함께 조명 리포트(illumination report)가 출력됩니다 — 힙 할당, 연산 비용, 메모리 전송, 빌림 상태가
소스 코드의 해당 줄 옆에 `INFO:`와 `WARN:`으로 표시됩니다.

이 문서는 wide로 프로그램을 작성하는 데 필요한 전부를 다룹니다. 모든 예제는 실제로 실행해
검증되었습니다.

## 목차

1. [시작하기](#1-시작하기)
2. [값과 변수](#2-값과-변수)
3. [연산자](#3-연산자)
4. [제어 흐름](#4-제어-흐름)
5. [함수와 클로저](#5-함수와-클로저)
6. [문자열](#6-문자열)
7. [컬렉션](#7-컬렉션)
8. [구조체, 열거형, 패턴 매칭](#8-구조체-열거형-패턴-매칭)
9. [에러 처리](#9-에러-처리)
10. [모듈](#10-모듈)
11. [입출력](#11-입출력)
12. [메모리와 빌림](#12-메모리와-빌림)
13. [텐서와 자동 미분](#13-텐서와-자동-미분)
14. [실행 백엔드와 성능](#14-실행-백엔드와-성능)
15. [치트시트](#15-치트시트)

---

## 1. 시작하기

```bash
cargo run -- program.wide          # 프로그램 실행
cargo run -- --vm program.wide     # 바이트코드 VM으로 실행
cargo test                         # 테스트 스위트
```

실행 순서는 항상 같습니다: **정적 검사 → 실행 → 조명 리포트**. 정적 검사가 에러를 찾으면
(미정의 이름, 인자 수 불일치, 텐서 모양 불일치, 확정적인 빌림 충돌 등) 프로그램은 실행되지
않고 에러 전체가 한 번에 보고됩니다.

```
# 주석은 # 으로 시작합니다
print("hello, wide")
```

문장은 줄바꿈으로 구분합니다. 세미콜론은 없습니다. 괄호 `(...)`와 대괄호 `[...]` 안에서는
줄바꿈이 공백으로 취급되므로 여러 줄에 걸친 리터럴과 인자 목록을 자유롭게 쓸 수 있습니다.

## 2. 값과 변수

```
n = 42                  # int
x = 3.14                # float (f32)
ok = true               # bool
s = "text"              # str
xs = [1, 2, 3]          # 배열
m = map{}               # 맵
r = 0..10               # 범위 (반열림: 0 이상 10 미만)
```

변수는 대입으로 만들어집니다. 타입은 동적으로 결정되며, `x: int = 5`처럼 타입을 표기할 수
있습니다(문서화 목적 — 텐서 모양 표기는 예외적으로 컴파일 타임에 검사됩니다. 13장 참조).

배열, 맵, 구조체, 텐서는 **참조 의미론**을 따릅니다: 변수에 대입하거나 함수에 넘겨도 같은
데이터를 공유합니다. 스칼라(int, float, bool)와 문자열은 값으로 복사됩니다.

## 3. 연산자

우선순위가 낮은 것부터:

```
or  and  not             # 논리 (단락 평가, bool 전용)
== != < > <= >=          # 비교
..                       # 범위
+ -                      # 덧셈·뺄셈 (+ 는 문자열·배열 연결도)
* /                      # 곱셈·나눗셈
-x   not x   &x   *p     # 단항
f(x)  x.m()  x.f  xs[i]  e?   # 후위
```

정수끼리의 나눗셈은 정수 나눗셈이며, 0으로 나누면 실행 시간 에러입니다(경고가 함께
조명됩니다).

## 4. 제어 흐름

```
if x < 0 {
    print("negative")
} elif x == 0 {
    print("zero")
} else {
    print("positive")
}

while cond {
    ...
    break       # 루프 종료
    continue    # 다음 반복
}

for i in 0..n { ... }       # 범위 순회
for x in xs { ... }         # 배열 순회
for ch in "abc" { ... }     # 문자열 순회 (문자 단위)
```

조건은 반드시 bool이어야 합니다. 0이나 빈 문자열이 자동으로 거짓이 되지 않습니다.

배열을 `for`로 순회하는 동안 그 배열은 공유 빌림 상태가 됩니다 — 루프 안에서 같은 배열을
변형하면(iterator invalidation) 충돌로 차단됩니다. 12장에서 자세히 다룹니다.

## 5. 함수와 클로저

```
fn add(a, b) {
    return a + b
}

fn fib(n) {                          # 재귀 (상호 재귀도 가능, 정의 순서 무관)
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}
```

함수는 전역 변수와 자신의 매개변수·지역 변수만 봅니다(호출한 쪽의 지역 변수는 보이지
않습니다).

함수는 값이기도 합니다:

```
add = fn(a, b) { return a + b }      # 익명 함수
print(add(2, 3))                     # 5

d = add                              # 함수 값 대입
fn apply(f, v) { return f(v) }       # 함수를 인자로

fn make_adder(n) {                   # 함수를 반환 — n을 캡처하는 클로저
    return fn(x) { return x + n }
}
add5 = make_adder(5)
print(add5(100))                     # 105

xs = [1, 2, 3, 4]
print(xs.map(fn(x) { return x * x }))     # [1, 4, 9, 16] — 새 배열
print(xs.filter(fn(x) { return x > 2 }))  # [3, 4]
```

클로저는 **생성 시점의 값**을 캡처합니다. 스칼라는 복사되므로 이후의 재대입은 클로저에
반영되지 않습니다. 배열·맵·구조체는 참조 공유이므로, 캡처된 배열에 클로저 안에서 push하면
바깥에서도 보입니다.

## 6. 문자열

```
s = "hello"
s.len                    # 5
s[1]                     # "e" (문자 단위 인덱싱)
s[-1]                    # "o" (음수는 끝에서)
s[1..4]                  # "ell" (슬라이스 — 복사)
s + ", world"            # 연결

s.upper()  s.lower()  s.trim()
s.split(",")             # 배열로 분리
s.chars()                # 문자 배열
s.contains("ell")  s.starts_with("he")  s.ends_with("lo")
s.replace("l", "L")  s.find("ll")
```

문자열은 불변입니다. 루프에서 문자열을 누적할 때는 `s = s + c`(매번 복사, O(n²)) 대신
가변 빌더를 사용하세요:

```
b = strbuf()             # 가변 문자열 빌더 (분할상환 O(1) append)
b.push("a")
b.push("bc")
s = b.str()              # "abc"
b.clear()
```

## 7. 컬렉션

### 배열

배열은 가변이며 스택·큐·덱을 겸합니다:

```
xs = [3, 1, 4]
xs[0]        xs[-1]        xs[1..3]         # 읽기 / 끝에서 / 슬라이스(복사)
xs[0] = 9                                   # 쓰기
xs.push(1)   xs.pop()                       # 스택
xs.push_front(0)   xs.pop_front()           # 큐·덱
xs.insert(1, 99)   xs.remove(0)
xs.sort()   xs.reverse()   xs.clear()
xs.len   xs.sum()   xs.contains(4)   xs.join(", ")
xs.map(f)   xs.filter(f)                    # 고차 메서드 (새 배열)
```

슬라이스는 항상 **복사**이며 범위를 벗어나면 에러입니다(자동으로 잘라내지 않습니다).
슬라이스에 대입할 수는 없습니다.

### 맵

```
m = map{}
m["k"] = 1                    # 삽입·갱신
m["k"]                        # 조회 (없는 키는 에러)
m.get("k", 0)                 # 기본값과 함께 조회
m.contains("k")   m.remove("k")
m.keys()   m.values()   m.len
```

맵 키는 int, str, bool만 허용됩니다. 키 순서는 결정적(정렬)입니다.

### 힙과 집합

우선순위 큐와 집합은 표준 모듈에 있습니다:

```
import "std/heap"
h = heap()
h.push(5)   h.push(1)
h.pop()                       # 1 (최소 힙)
h.peek()

import "std/set"
s = set()
s.add(3)   s.contains(3)   s.remove(3)   s.items()
```

## 8. 구조체, 열거형, 패턴 매칭

```
struct Point { x, y }

p = Point { x: 1, y: 2 }
p.x = 10                      # 필드 대입 (참조 의미론 — 공유한 쪽 모두에 보임)

impl Point {
    fn dist2(self) { return self.x * self.x + self.y * self.y }
    fn shift(self, dx) { self.x = self.x + dx }    # self 변경이 호출자에게 보임
}
p.dist2()
p.shift(3)
```

메서드의 첫 매개변수는 명시적인 `self`입니다.

```
enum Shape {
    Circle(r)
    Rect(w, h)
    Dot
}

s = Shape::Circle(5)

# 문장 match — 갈래는 블록
match s {
    Shape::Circle(r) => { print("circle", r) }
    Shape::Rect(w, h) => { print("rect", w * h) }
    Shape::Dot => { print("dot") }
}

# 식 match — 갈래는 표현식 (콤마 또는 줄바꿈으로 구분)
area = match s {
    Shape::Circle(r) => 3 * r * r,
    Shape::Rect(w, h) => w * h,
    _ => 0
}
```

패턴에는 리터럴, 와일드카드 `_`, 바인딩, 열거형 변형, 구조체 패턴(`Point { x: 0, y }`)을
쓸 수 있습니다. 재귀적 열거형이 가능하므로 연결 리스트나 트리 같은 자료구조를 직접 정의할
수 있습니다.

### 클래스

`class`는 구조체와 impl을 하나로 합친 선언입니다. 필드를 먼저 쓰고 메서드를 씁니다.
`self`가 없는 메서드는 **연관 함수**로, `이름::함수(...)` 형태로 호출합니다 — 관례상 `new`가
생성자입니다:

```
class Counter {
    n, step

    fn new(start) {                      # 연관 함수 (self 없음) — 생성자
        return Counter { n: start, step: 1 }
    }

    fn tick(self) {                      # 인스턴스 메서드
        self.n = self.n + self.step
        return self.n
    }
}

c = Counter::new(10)
c.tick()                                 # 11
c.n                                      # 11
```

클래스는 곧 구조체이기도 하므로 `Counter { n: 0, step: 2 }`처럼 리터럴로 직접 만들 수도
있고, 패턴 매칭에도 그대로 쓸 수 있습니다.

## 9. 에러 처리

wide는 예외가 아니라 **에러 값**을 사용합니다. 함수는 정상 값 또는 에러 값을 반환하고,
호출자가 처리 방법을 선택합니다:

```
fn safe_div(a, b) {
    if b == 0 { return err("division by zero") }
    return a / b
}

# 방법 1: 직접 검사
r = safe_div(10, 0)
if is_err(r) {
    print(err_msg(r))         # "division by zero"
}

# 방법 2: ? 로 전파 — 에러면 현재 함수가 그 에러를 그대로 반환
fn compute(a, b) {
    q = safe_div(a, b)?
    return q + 1
}
```

`?`는 체인의 어느 단계에서 실패했든 에러를 위로 올려 보냅니다. 파일 입출력(11장)의 실패도
같은 에러 값으로 돌아오므로 처리 방식이 하나로 통일됩니다.

## 10. 모듈

```
import "lib/util.wide"        # 상대 경로
```

import된 파일의 함수·구조체·열거형은 현재 파일에서 바로 사용할 수 있으며, 가시성은
전이적입니다(간접 import 포함). 중복 import와 순환 import는 안전하게 처리됩니다.

표준 모듈은 `std/` 접두로 활성화합니다: `import "std/ai"`(텐서), `import "std/fs"`(파일),
`import "std/heap"`, `import "std/set"`. 활성화하지 않고 해당 기능을 쓰면 정적 검사가
어느 import가 필요한지 알려줍니다.

## 11. 입출력

### 콘솔

```
print(a, b)                   # 공백으로 구분해 출력, 줄바꿈 자동

cout << "name: " << name << "\n"    # 스트림 출력 — 자동 공백 없음, 수동 제어
cin >> x >> y                       # 입력 — 공백 단위로 나눠 자동 타입 추론
                                    # (정수→int, 소수→float, 그 외→str)
```

`cin`의 대상은 변수, 배열 원소(`xs[0]`), 필드(`p.x`) 모두 가능합니다.

### 파일

```
import "std/fs"

write_file("notes.txt", "alpha\nbeta")    # 생성 또는 덮어쓰기
s = read_file("notes.txt")                # 전체를 문자열로
lines = read_lines("notes.txt")           # 줄 단위 배열
append_file("notes.txt", "\ngamma")
file_exists("notes.txt")                  # bool
remove_file("notes.txt")

r = read_file("missing.txt")              # 실패는 에러 값 — 프로그램이 죽지 않음
if is_err(r) { print(err_msg(r)) }
s = read_file(path)?                      # 함수 안에서는 ? 로 전파
```

모든 읽기/쓰기는 바이트 수가 조명됩니다 — 입출력 비용은 숨겨지지 않습니다.

## 12. 메모리와 빌림

wide의 메모리 접근 설계는 한 문장으로 요약됩니다: **막지 말고, 보여줘라.**

### 포인터

```
xs = [10, 20, 30, 40]
p = &xs[1]             # 원소의 주소 — 어디서 왔는지(provenance)가 조명됩니다
print(*p)              # 20
*p = 99                # 포인터를 통한 쓰기 (경계 검사됨)
```

경계 밖 주소(`&xs[9]`)는 차단되고, 배열이 줄어들어 무효해진 포인터는 역참조가 거부됩니다.
추적 중인 레코드는 언제든 조회할 수 있습니다:

```
@show provenance p
# INFO: provenance: origin xs · extent 0..4 · pointee [1] · alive true · access owner
```

### 검사 없는 접근 — raw

일반 접근은 위반 시 차단하지만, `raw.*`는 같은 추적을 유지한 채 위험을 **조명만 하고
진행**합니다. 책임이 호출자에게 넘어갑니다:

```
ps = &xs[0]
raw.read(ps, 3)          # INFO: 범위 안 — 안전 확인됨
raw.read(ps, 6)          # WARN: 범위 초과 — 오버런 가능 (차단하지 않음)
raw.write(ps, [1, 2])
raw.memcpy(dst, ps, 4)   # 양쪽 범위와 대조해 조명
```

### 빌림 — 증명, 가드, 신뢰의 그라데이션

빌림 규칙은 하나입니다: **쓰기 빌림 하나 XOR 읽기 빌림 여럿.** wide가 다른 언어와 다른
점은 이 규칙을 지키는 *방법*이 연속체라는 것입니다:

```
r = &xs                # 공유 빌림 — 이 스코프가 끝날 때까지 xs 변형은 충돌
m = &mut xs            # 배타 빌림 — 다른 빌림과 순회가 충돌
```

컴파일러가 빌림이 안전하다고 **증명**하면 런타임 가드조차 붙지 않습니다(비용 0):

```
r = &xs
print(r[0])            # INFO: shared borrow of xs statically proven safe — cost 0
```

확실한 충돌은 실행 전에 컴파일 에러로 잡힙니다:

```
r = &xs
xs.push(2)             # 컴파일 에러: borrow conflict (caught before run)
```

증명할 수 없는 경우(변수가 함수로 넘어가거나, 별칭이 생기는 등)에는 자동으로 **런타임
가드**로 내려가며, 조건문 안의 잠재적 충돌은 절대 컴파일 에러가 되지 않습니다 — 실행되지
않을 수도 있는 코드로 정상 프로그램을 거부하지 않습니다(거짓 양성 0). 마지막으로, 검사를
직접 끄고 싶다면:

```
@trust xs.push(9)      # WARN: 이 문장만 검사 해제 — 책임은 작성자
```

Rust가 "컴파일에 통과하거나, 포기하거나"의 양자택일이라면, wide에서는 증명하지 못한 코드도
가드와 함께 그대로 실행됩니다. 어느 단계가 적용됐는지는 항상 조명으로 확인할 수 있습니다.

## 13. 텐서와 자동 미분

`import "std/ai"`로 활성화합니다. 설계 목표는 "들여다보이는 PyTorch" — 모든 연산이 모양,
바이트, FLOP, 전송, 활성 메모리를 조명합니다.

### 텐서 만들기와 연산

```
import "std/ai"

a = tensor([[1, 2, 3], [4, 5, 6]])     # 중첩 배열 → f32 텐서 (모양 추론)
zeros([2, 3])   ones([2, 3])
a.shape   a.size   a.ndim

a + 1     a * 2                        # 스칼라 브로드캐스트
a + b                                  # elementwise (NumPy 브로드캐스트 규칙)
matmul(a, w)                           # 행렬곱 (2차원)
conv2d(x, k)                           # 2차원 합성곱 (valid, stride 1)
maxpool2d(x, 2)                        # k×k 최대 풀링
relu(t)  sigmoid(t)  tanh(t)  exp(t)  log(t)  softmax(t)  transpose(t)

a.sum()   a.mean()   a.max()           # 환원 → 스칼라 텐서 (.item()으로 수 추출)
a.sum(0)  a.mean(1)                    # 축별 환원
a.reshape([3, 2])                      # 모양 변경 (원소 수 일치 필요)
```

### 모양은 컴파일 타임에 검사됩니다

행렬곱의 차원 불일치나 브로드캐스트 불가능한 조합은 프로그램이 실행되기 전에 잡힙니다.
기호 차원을 표기하면 관계까지 검사됩니다:

```
fn layer(x: tensor[(B, K)], w: tensor[(K, N)]) {
    return matmul(x, w)
}
# 호출 지점에서 두 K가 다른 크기로 묶이면 컴파일 에러
```

대문자는 기호 차원(관계 검사), 소문자는 동적 차원, `?`는 미지를 뜻합니다.

### 자동 미분과 학습

```
w = param([[0], [0]])          # 학습 파라미터 (기울기 추적)
pred = matmul(x, w)
diff = pred - target
loss = (diff * diff).mean()    # 손실은 스칼라여야 backward 가능
loss.backward()                # 역전파
w.grad                         # 누적 기울기
grad_step(w, 0.01)             # SGD 한 스텝 (기울기 리셋 포함)
adam_step(w, 0.01)             # Adam 한 스텝 (모멘트 상태는 텐서 안에 유지)
```

이 조합으로 회귀, 다층 퍼셉트론, 로지스틱/softmax 분류기, 합성곱 신경망까지 학습할 수
있습니다. `examples/ai/`에 동작하는 예제가 있습니다 — `cnn.wide`는 conv2d → relu →
maxpool2d → reshape → matmul 사슬을 끝까지 미분해 분류기를 학습합니다.

현재 한계: `.backward()`는 스칼라에서만, 나눗셈은 미분 불가, matmul은 2차원 전용입니다.

### 내장 모델과 EDA — std/ml

자주 쓰는 간단한 모델은 만들어져 있습니다. `import "std/ml"`로 가져오며, 모델 자체가 wide로
작성되어 있고 이 언어의 자동 미분으로 학습됩니다:

```
import "std/ml"

m = logistic_regression()
m.fit(x, y, 300, 0.1)          # x: (n,d), y: (n,1) 0/1 레이블 — 최종 BCE 손실 반환
m.predict(x)                   # 확률 텐서
m.score(x, y)                  # BCE (낮을수록 좋음)

lm = linear_regression()       # fit / predict / score(MSE) 동일 구성
```

데이터 탐색 도우미도 함께 제공됩니다:

```
t = read_csv("data.csv")       # 숫자 CSV → 텐서 (헤더 행은 감지해 건너뜀)
describe(t)                    # shape, 열별 mean/std, min/max 요약
sqrt(t)                        # 텐서 제곱근 (elementwise, 미분 가능)
```

`read_csv`의 실패는 파일 입출력과 같은 에러 값으로 돌아옵니다.

### GPU

```
cargo run --features gpu -- program.wide
```

`gpu` 피처를 켜고 빌드하면 `.gpu()`로 옮긴 텐서의 행렬곱과 elementwise 연산이 실제 GPU
컴퓨트 셰이더로 실행됩니다:

```
a = tensor([[...]]).gpu()      # 실제 업로드 — 조명에 어댑터 이름 표시
b = tensor([[...]]).gpu()
c = matmul(a, b) * 2 + 1       # 사슬 전체가 GPU에 머묾 — 재업로드 0
print(c.cpu())
```

전송은 전부 조명됩니다. 작은 행렬에서는 전송 오버헤드 때문에 CPU가 더 빠를 수 있습니다 —
그 교차점을 보여주는 것이 이 언어의 목적입니다. 측정치는 [BENCH.md](BENCH.md)에 있습니다.
피처를 켜지 않으면 같은 코드가 CPU에서 동일한 결과로 실행됩니다.

## 14. 실행 백엔드와 성능

| 백엔드 | 실행 방법 | 설명 |
|--------|-----------|------|
| 트리워커 | `cargo run -- f.wide` | 기본. 전 기능 지원(레퍼런스 구현) |
| 바이트코드 VM | `--vm` | 코어 언어 전체 지원. 클로저·빌림·텐서는 아직 트리워커 전용 |
| JIT | `--features jit` | 수치 함수(int/f64, 재귀 포함)를 기계어로 자동 컴파일 |
| GPU | `--features gpu` | 텐서 행렬곱·elementwise를 GPU 셰이더로 |

JIT은 적격인 함수를 자동으로 골라 컴파일하며(조명으로 표시), 적격이 아닌 함수는 그대로
인터프리터에서 실행됩니다 — 코드는 바꿀 필요가 없습니다. fib(30) 기준으로 인터프리터 대비
수백 배 빠릅니다. 정확한 수치와 측정 방법은 [BENCH.md](BENCH.md)를 참고하세요.

`--time` 플래그는 실행 시간을 stderr로 출력합니다(프로세스 기동 제외 — 벤치마크용).

## 15. 치트시트

```
# 변수            x = 5      x: int = 5
# 출력            print(a, b)       cout << a << "\n"
# 입력            cin >> x >> y                       # 자동 타입 추론
# 산술/비교        + - * /   == != < > <= >=
# 논리            and  or  not                        # 단락 평가
# 범위/인덱스      0..n   xs[i]   xs[-1]   xs[1..3]    # 슬라이스는 복사
# 조건            if c { } elif d { } else { }
# 반복            while c { }   for x in xs { }   break   continue
# 함수            fn f(a, b) { return a + b }
# 클로저          g = fn(x) { return x + k }   xs.map(f)   xs.filter(f)
# 문자열          "..."  .upper() .split(s) .len       # 누적은 strbuf()
# 배열            [1, 2]  .push .pop .sort .sum .len
# 맵              map{}   m[k] = v   .get(k, 기본값) .keys
# 구조체          struct P { x, y }   P { x: 1, y: 2 }   p.x
# 메서드          impl P { fn m(self) { ... } }   p.m()
# 열거형          enum E { A(v)  B }   E::A(5)
# 클래스          class C { 필드들  fn new(..) { }  fn m(self) { } }   C::new(..)
# 매칭            match v { E::A(x) => { ... }  _ => { ... } }
# 에러            err(m)  is_err(v)  err_msg(v)  f()?
# 모듈            import "file.wide"   import "std/ai" "std/fs" "std/heap" "std/set"
# 파일            read_file  read_lines  write_file  append_file  remove_file  file_exists
# 포인터          p = &xs[i]   *p   @show provenance p
# 빌림            r = &xs   m = &mut xs   @trust <문장>
# raw            raw.read(p, n)  raw.write(p, vals)  raw.memcpy(d, s, n)
# 텐서            tensor  param  zeros  ones  matmul  conv2d  maxpool2d  relu  softmax
# 자동미분         loss.backward()   w.grad   grad_step(w, lr)   adam_step(w, lr)
```
