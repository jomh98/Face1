import base64
import hashlib
import io
import itertools
import os
import warnings
from collections.abc import Iterator

import openai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MODEL = "gpt-4o"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_PIXELS = 40_000_000  # 약 7700x5200. 이보다 큰 이미지는 디코딩 전에 거부한다(메모리 보호).
MAX_SIDE_PX = 1024
JPEG_QUALITY = 85
# 스트리밍에서 timeout은 "응답 바이트 사이의 최대 대기 시간"이다. 재시도 1회 포함 최악 약 60초.
API_TIMEOUT_SEC = 30
API_MAX_RETRIES = 1
MAX_CACHE_ENTRIES = 20
MAX_ANALYSES_PER_SESSION = 10  # 공개 배포 시 API 비용 남용 방지 (새로고침하면 초기화됨)
PREVIEW_WIDTH_PX = 280

DISCLAIMER = "본 결과는 재미를 위한 것이며 과학적 근거가 없습니다."
PRIVACY_NOTICE = "이 앱은 사진을 저장하지 않으며, 분석을 위해 사진이 OpenAI로 전송됩니다."
NO_FACE_MESSAGE = (
    "사진에서 한 사람의 얼굴을 또렷하게 확인하기 어렵습니다. "
    "정면을 바라본 한 사람의 얼굴이 잘 보이는 사진으로 다시 시도해 주세요."
)

SYSTEM_PROMPT = f"""당신은 전통 관상학에 정통한 친근한 관상가입니다.
사용자가 올린 얼굴 사진을 보고 재미로 즐길 수 있는 관상 풀이를 한국어로 작성합니다.

규칙:
- 사진 속 인물이 누구인지 절대 추측하거나 식별하지 마세요. 유명인과 닮았다는 언급도 하지 마세요.
- 나이, 인종, 성별 등 민감한 속성을 단정하지 말고 얼굴 생김새의 특징만 근거로 풀이하세요.
- 전체적으로 긍정적이고 따뜻한 톤을 유지하세요. 약점도 조언의 형태로 부드럽게 표현하세요.
- 사진에 사람의 얼굴이 없거나, 2명 이상이 있거나, 얼굴을 알아보기 어려우면
  다른 말 없이 다음 문장만 그대로 출력하세요:
  {NO_FACE_MESSAGE}

출력 형식 (Markdown, 아래 제목을 그대로 사용):

### 🌟 전체 인상
한 줄 요약

### 🔍 부위별 풀이
- **이마**: ...
- **눈썹**: ...
- **눈**: ...
- **코**: ...
- **입**: ...
- **귀**: ...
- **얼굴형**: ...

### 🍀 운세 풀이
- **재물운**: ...
- **애정운**: ...
- **직업운**: ...
- **건강운**: ...

### 📜 종합 조언
긍정적인 한마디

귀가 머리카락 등에 가려 보이지 않으면 "가려져 있어 보기 어렵지만"이라고 밝힌 뒤 전체 인상에 기반해 짧게 풀이하세요.
"""

USER_PROMPT = "이 사진 속 얼굴의 관상을 풀이해 주세요."

# 테마 색/폰트는 .streamlit/config.toml에서 관리하고, 여기서는 레이아웃 보정만 한다.
CUSTOM_CSS = """
<style>
/* 상단 고정 툴바(약 60px)에 헤더가 가리지 않도록 위쪽 여백을 충분히 둔다. */
.block-container { max-width: 1120px; padding-top: 4.5rem; }
.app-header { text-align: center; margin-bottom: 1.75rem; }
.app-header .seal {
  display: inline-block; padding: .2rem .45rem; margin-bottom: .6rem;
  border: 2px solid #B3261E; border-radius: .35rem; color: #B3261E;
  font-family: "Nanum Myeongjo", serif; font-weight: 800; font-size: 1.05rem;
  letter-spacing: .15em; transform: rotate(-4deg);
}
.app-header h1 { padding: 0 0 .35rem; letter-spacing: .04em; }
.app-header p { margin: 0; opacity: .8; }
.app-header .small { font-size: .85rem; opacity: .6; margin-top: .35rem; }
</style>
"""

HEADER_HTML = """
<div class="app-header">
  <div class="seal">觀相</div>
  <h1>AI 관상가</h1>
  <p>얼굴 사진 한 장으로 보는 전통 관상 풀이</p>
  <p class="small">※ {disclaimer}</p>
</div>
"""


class ImageError(Exception):
    """사용자에게 그대로 보여줄 수 있는 이미지 처리 오류."""


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def get_config() -> dict:
    """API 키와 모델명을 st.secrets → .env 순서로 읽는다."""
    load_dotenv()

    def read(key: str) -> str | None:
        try:
            value = st.secrets.get(key)
            if value:
                return str(value)
        except Exception:
            # 로컬에 secrets.toml이 없으면 st.secrets 접근 시 예외가 발생한다.
            pass
        return os.getenv(key) or None

    return {
        "api_key": read("OPENAI_API_KEY"),
        "model": read("OPENAI_MODEL") or DEFAULT_MODEL,
    }


@st.cache_resource
def get_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, timeout=API_TIMEOUT_SEC, max_retries=API_MAX_RETRIES)


# ---------------------------------------------------------------------------
# 이미지 전처리
# ---------------------------------------------------------------------------
def preprocess_image(data: bytes) -> bytes:
    """크기 검사 → EXIF 회전 보정 → RGB 변환 → 축소 → JPEG 바이트."""
    if len(data) > MAX_FILE_BYTES:
        raise ImageError("파일 크기가 10MB를 초과합니다. 더 작은 사진을 올려 주세요.")

    too_large = ImageError(
        "이미지 해상도가 너무 큽니다. 가로·세로 크기가 더 작은 사진을 올려 주세요."
    )
    try:
        # open()은 헤더만 읽으므로, 픽셀 데이터를 메모리에 올리기 전에 해상도를 검사할 수 있다.
        # 해상도는 아래에서 직접 검사하므로 Pillow의 DecompressionBombWarning은 끈다.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
        width, height = img.size
        if width * height > MAX_PIXELS:
            raise too_large
        # JPEG는 디코딩 단계에서 축소해 읽어 메모리 사용을 줄인다. (다른 형식은 무시됨)
        img.draft("RGB", (MAX_SIDE_PX, MAX_SIDE_PX))
        img.load()
    except Image.DecompressionBombError:
        raise too_large
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise ImageError("이미지를 읽을 수 없습니다. jpg, png, webp 형식의 사진을 올려 주세요.")

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def image_hash(jpeg_bytes: bytes, model: str) -> str:
    return hashlib.sha256(model.encode() + b"\0" + jpeg_bytes).hexdigest()


# ---------------------------------------------------------------------------
# OpenAI 분석
# ---------------------------------------------------------------------------
def stream_reading(client: openai.OpenAI, model: str, data_url: str) -> Iterator[str]:
    """API 요청을 즉시 보내고(연결/인증 오류는 여기서 발생) 텍스트 조각 이터레이터를 반환한다."""
    stream = client.chat.completions.create(
        model=model,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return _iter_text(stream)


def _iter_text(stream) -> Iterator[str]:
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "refusal", None):
            yield delta.refusal
        if delta.content:
            yield delta.content


def describe_api_error(err: Exception) -> str:
    # APITimeoutError는 APIConnectionError의 하위 클래스이므로 먼저 검사한다.
    if isinstance(err, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. OPENAI_API_KEY 설정을 확인해 주세요."
    if isinstance(err, openai.RateLimitError):
        return "요청이 너무 많거나 API 사용 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(err, openai.APITimeoutError):
        return "분석 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(err, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인한 뒤 다시 시도해 주세요."
    if isinstance(err, openai.NotFoundError):
        return "설정된 모델을 찾을 수 없습니다. OPENAI_MODEL 값을 확인해 주세요."
    if isinstance(err, openai.BadRequestError):
        return "요청이 거부되었습니다. 다른 사진으로 시도하거나 이미지 입력을 지원하는 모델인지 확인해 주세요."
    if isinstance(err, openai.APIError):
        return "OpenAI 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    return "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


# ---------------------------------------------------------------------------
# 결과 캐시 (세션별)
# ---------------------------------------------------------------------------
# st.cache_data는 함수 반환값을 캐싱하므로 스트리밍(제너레이터)과 함께 쓸 수 없다.
# 그래서 "캐시에 있으면 바로 표시, 없으면 스트리밍 후 저장" 구조를 세션 상태로 구현한다.
# 세션별로 두어 다른 사용자의 사진 결과가 공유되지 않게 한다.
def cache_get(key: str) -> str | None:
    return st.session_state.setdefault("reading_cache", {}).get(key)


def cache_put(key: str, value: str) -> None:
    cache: dict = st.session_state.setdefault("reading_cache", {})
    cache[key] = value
    while len(cache) > MAX_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _widget_gen() -> int:
    # 파일 업로더/카메라 값은 코드로 지울 수 없으므로, 초기화할 때 위젯 key를 바꾼다.
    return st.session_state.get("widget_gen", 0)


def _set_source(source: str) -> None:
    st.session_state["image_source"] = source


def reset_app() -> None:
    # 분석 횟수(api_calls)는 일부러 유지한다.
    st.session_state["widget_gen"] = _widget_gen() + 1
    for key in ("image_source", "last_result", "prepared"):
        st.session_state.pop(key, None)


def remaining_analyses() -> int:
    return MAX_ANALYSES_PER_SESSION - st.session_state.get("api_calls", 0)


def prepare_image(data: bytes) -> bytes:
    """전처리 결과를 세션에 보관해, 화면이 다시 그려질 때마다 재처리하지 않게 한다."""
    raw_key = hashlib.sha256(data).hexdigest()
    prepared = st.session_state.get("prepared")
    if prepared and prepared[0] == raw_key:
        return prepared[1]
    jpeg_bytes = preprocess_image(data)  # ImageError는 호출자가 처리
    st.session_state["prepared"] = (raw_key, jpeg_bytes)
    return jpeg_bytes


def render_sidebar(config: dict) -> None:
    with st.sidebar:
        st.header("📖 사용 방법")
        st.markdown(
            "1. **사진 업로드** 또는 **카메라 촬영** 탭에서 얼굴 사진을 준비합니다.\n"
            "2. **🔮 관상 보기** 버튼을 누릅니다.\n"
            "3. **📜 관상 풀이** 영역에 나타나는 결과를 확인합니다."
        )
        st.caption("정면을 바라본 한 사람의 얼굴이 밝게 나온 사진일수록 풀이가 정확해집니다.")
        st.divider()
        st.subheader("🔒 개인정보")
        st.caption(f"{PRIVACY_NOTICE} 결과는 현재 브라우저 세션에만 잠시 보관됩니다.")
        st.subheader("⚠️ 안내")
        st.caption(DISCLAIMER)
        st.divider()
        st.caption(f"사용 모델: `{config['model']}`")


def get_input_image():
    """업로드/카메라 중 가장 최근에 입력된 이미지를 반환한다."""
    gen = _widget_gen()
    tab_upload, tab_camera = st.tabs(["📁 사진 업로드", "📷 카메라 촬영"])
    with tab_upload:
        uploaded = st.file_uploader(
            "얼굴 사진을 업로드하세요 (jpg, png, webp · 최대 10MB)",
            type=ALLOWED_TYPES,
            key=f"upload_{gen}",
            on_change=_set_source,
            args=("upload",),
        )
    with tab_camera:
        captured = st.camera_input(
            "카메라로 촬영하세요",
            key=f"camera_{gen}",
            on_change=_set_source,
            args=("camera",),
        )
    st.caption(f"🔒 {PRIVACY_NOTICE}")

    images = {"upload": uploaded, "camera": captured}
    preferred = st.session_state.get("image_source", "upload")
    if images.get(preferred) is not None:
        return images[preferred]
    return uploaded or captured


def run_analysis(config: dict, jpeg_bytes: bytes) -> str | None:
    """캐시에 있으면 바로 표시, 없으면 스트리밍 후 저장. 성공 시 결과 텍스트를 반환한다."""
    key = image_hash(jpeg_bytes, config["model"])
    cached = cache_get(key)
    if cached is not None:
        st.markdown(cached)
        return cached

    if remaining_analyses() <= 0:
        st.warning(f"이번 세션의 분석 횟수({MAX_ANALYSES_PER_SESSION}회)를 모두 사용했습니다.")
        return None

    st.session_state["api_calls"] = st.session_state.get("api_calls", 0) + 1
    client = get_client(config["api_key"])
    output = st.empty()
    try:
        # 스피너는 첫 글자가 도착할 때까지만 보여주고, 이후에는 결과가 바로 흘러나오게 한다.
        with st.spinner("관상을 살펴보는 중입니다..."):
            chunks = stream_reading(client, config["model"], to_data_url(jpeg_bytes))
            first = next(chunks, "")
        text = output.write_stream(itertools.chain([first], chunks))
    except Exception as err:
        # 중간에 끊겨도 반쯤 나온 풀이를 남기지 않는다.
        output.empty()
        st.error(describe_api_error(err))
        return None

    if not (isinstance(text, str) and text.strip()):
        output.empty()
        st.warning("결과를 받지 못했습니다. 다시 시도해 주세요.")
        return None

    # "얼굴 확인 어려움"은 캐시하지 않아, 같은 사진으로도 다시 시도할 수 있게 한다.
    if text.strip() != NO_FACE_MESSAGE:
        cache_put(key, text)
    return text


def render_result(config: dict, jpeg_bytes: bytes | None, clicked: bool) -> None:
    st.subheader("📜 관상 풀이")
    key = image_hash(jpeg_bytes, config["model"]) if jpeg_bytes else None
    last = st.session_state.get("last_result")

    if clicked and jpeg_bytes:
        with st.container(border=True):
            text = run_analysis(config, jpeg_bytes)
        if text is not None:
            st.session_state["last_result"] = {"key": key, "text": text}
        has_result = text is not None
    elif key and last and last["key"] == key:
        # 다른 위젯 조작으로 재실행돼도 같은 사진의 결과는 계속 보여준다.
        with st.container(border=True):
            st.markdown(last["text"])
        has_result = True
    else:
        with st.container(border=True):
            if jpeg_bytes:
                st.markdown("사진이 준비되었습니다. **🔮 관상 보기** 버튼을 눌러 주세요.")
            else:
                st.markdown("얼굴 사진을 올리고 **🔮 관상 보기** 버튼을 누르면 이곳에 관상 풀이가 나타납니다.")
        has_result = False

    if has_result:
        st.caption(f"※ {DISCLAIMER}")
        st.button("🔄 다른 사진으로 다시 보기", on_click=reset_app, width="stretch")


def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    config = get_config()
    render_sidebar(config)
    st.markdown(HEADER_HTML.format(disclaimer=DISCLAIMER), unsafe_allow_html=True)

    if not config["api_key"]:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다. "
            "로컬에서는 `.env` 파일에, Streamlit Cloud에서는 앱 Secrets에 "
            "`OPENAI_API_KEY`를 설정해 주세요."
        )
        st.stop()

    left, right = st.columns([5, 6], gap="large")

    with left:
        st.subheader("🖼️ 사진 준비")
        image_file = get_input_image()

        jpeg_bytes = None
        if image_file is not None:
            try:
                jpeg_bytes = prepare_image(image_file.getvalue())
            except ImageError as err:
                st.error(str(err))
            else:
                # 세로 사진이 버튼을 화면 밖으로 밀어내지 않도록 너비를 제한한다.
                st.image(jpeg_bytes, caption="분석할 사진", width=PREVIEW_WIDTH_PX)

        remaining = remaining_analyses()
        is_cached = jpeg_bytes is not None and cache_get(
            image_hash(jpeg_bytes, config["model"])
        ) is not None
        clicked = st.button(
            "🔮 관상 보기",
            type="primary",
            disabled=jpeg_bytes is None or (remaining <= 0 and not is_cached),
            width="stretch",
        )
        # 분석(오른쪽 컬럼)이 끝난 뒤의 횟수를 보여주기 위해 자리만 먼저 잡는다.
        count_slot = st.empty()

    with right:
        render_result(config, jpeg_bytes, clicked)

    remaining = remaining_analyses()
    if remaining <= 0:
        count_slot.caption(
            f"이번 세션의 분석 횟수({MAX_ANALYSES_PER_SESSION}회)를 모두 사용했습니다. "
            "이미 본 사진의 결과는 다시 볼 수 있습니다."
        )
    else:
        count_slot.caption(f"남은 분석 횟수: {remaining}회")


if __name__ == "__main__":
    main()
