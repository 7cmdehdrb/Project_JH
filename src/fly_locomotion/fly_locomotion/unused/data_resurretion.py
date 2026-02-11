import re
from pathlib import Path

LABELS = {"one", "two", "unknown", "translation"}

# 숫자(부호/소수/지수 포함) + 다음 숫자의 부호(+/-)가 바로 붙은 경우를 찾아 분리
# 단, 지수 표기(e-3, E+5)의 +/-는 분리하면 안 되므로 e/E 뒤의 부호는 제외한다.
_FIX_JOINED_NUMBERS_RE = re.compile(r"(?<![eE])([0-9.])([+-])")


def fix_line(line: str) -> str:
    line = line.strip()
    if not line:
        return line

    # 첫 토큰이 라벨인지 확인
    parts = line.split(maxsplit=1)
    if len(parts) == 1:
        return line

    label, rest = parts[0], parts[1]
    if label not in LABELS:
        # 라벨이 아닌 줄도 같은 방식으로 고칠 수 있게 전체를 대상으로 처리
        return _FIX_JOINED_NUMBERS_RE.sub(r"\1 \2", line)

    # 숫자 부분만 고친다
    fixed_rest = _FIX_JOINED_NUMBERS_RE.sub(r"\1 \2", rest)
    # 여러 공백 정리
    fixed_rest = re.sub(r"\s+", " ", fixed_rest).strip()
    return f"{label} {fixed_rest}"


def fix_file(in_path: str, out_path: str) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    with in_path.open("r", encoding="utf-8", errors="ignore") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            fout.write(fix_line(line) + "\n")


if __name__ == "__main__":
    # 사용 예:
    fix_file(
        "/home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/dataskeleton_data_shuffled.txt",
        "/home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/dataskeleton_data_shuffled_fixed.txt",
    )
    pass
