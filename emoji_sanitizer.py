"""Streamlit emoji/mojibake sanitizer."""

from __future__ import annotations

import re
from functools import wraps
from typing import Any


class EmojiSanitizer:
    """Sanitize Streamlit UI text by fixing mojibake and stripping emoji."""

    _EMOJI_RE = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "\U0001F1E6-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )

    def __init__(self, st_module):
        self._st = st_module

    def _fix_mojibake(self, text: str) -> str:
        if "ð" not in text and "â" not in text:
            return text
        try:
            return text.encode("latin-1").decode("utf-8")
        except UnicodeError:
            return text

    def _strip_emoji(self, text: str) -> str:
        text = self._fix_mojibake(text)
        return self._EMOJI_RE.sub("", text)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._strip_emoji(value)
        if isinstance(value, list):
            return [self._strip_emoji(v) if isinstance(v, str) else v for v in value]
        return value

    def _wrap(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if args:
                args = (self._sanitize_value(args[0]),) + args[1:]

            for key in ("label", "value", "help", "placeholder"):
                if key in kwargs:
                    kwargs[key] = self._sanitize_value(kwargs[key])

            return fn(*args, **kwargs)

        return wrapper

    def patch(self) -> None:
        if getattr(self._st, "_emoji_patched", False):
            return

        methods = [
            "title",
            "header",
            "subheader",
            "markdown",
            "write",
            "text",
            "caption",
            "info",
            "warning",
            "error",
            "success",
            "button",
            "download_button",
            "selectbox",
            "multiselect",
            "radio",
            "checkbox",
            "text_input",
            "number_input",
            "slider",
            "expander",
            "tabs",
        ]

        for name in methods:
            if hasattr(self._st, name):
                setattr(self._st, name, self._wrap(getattr(self._st, name)))

        if hasattr(self._st, "sidebar"):
            for name in methods:
                if hasattr(self._st.sidebar, name):
                    setattr(self._st.sidebar, name, self._wrap(getattr(self._st.sidebar, name)))

        self._st._emoji_patched = True
