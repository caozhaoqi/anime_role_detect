#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""

import re
from .constants import (
    NON_CHARACTER_KEYWORDS,
    CHARACTER_TYPE_KEYWORDS,
    ALLOWED_SYMBOLS,
    LENGTH_MIN,
    LENGTH_MAX,
    MAX_SPACES,
    MAX_HYPHENS,
    MAX_DOTS,
    CHINESE_PATTERN,
    FULL_WIDTH_ALPHA_NUM_PATTERN,
    HALF_WIDTH_ALPHA_NUM_PATTERN,
    VERSION_PATTERN,
    DATE_PATTERN,
    TIME_PATTERN,
    URL_PATTERN,
    EMAIL_PATTERN,
    PATH_PATTERN,
    COMMAND_PATTERN,
    CODE_PATTERN,
    TAG_PATTERN,
    BRACKET_PATTERN,
    EMOJI_PATTERN,
    SPECIAL_CHAR_PATTERN,
    ENGLISH_WORD_PATTERN,
    ALPHA_NUM_PATTERN,
    REPEAT_PATTERN,
    CONTROL_CHAR_PATTERN,
    FULL_WIDTH_PATTERN,
    NON_ASCII_PATTERN,
    ASCII_PATTERN,
    DIGIT_PATTERN,
    ALPHA_PATTERN,
    MIXED_PATTERN,
    REPEAT_CHAR_PATTERN,
    SPACES_PATTERN,
    WHITESPACE_PATTERN,
    INVISIBLE_PATTERN,
    MARKDOWN_PATTERN,
    MARKDOWN_LINK_PATTERN,
    MARKDOWN_IMAGE_PATTERN,
    MARKDOWN_HEADING_PATTERN,
    MARKDOWN_LIST_PATTERN,
    MARKDOWN_QUOTE_PATTERN,
    MARKDOWN_CODE_PATTERN,
    MARKDOWN_BOLD_PATTERN,
    MARKDOWN_ITALIC_PATTERN,
    MARKDOWN_STRIKETHROUGH_PATTERN,
    MARKDOWN_INLINE_CODE_PATTERN,
    MARKDOWN_TABLE_PATTERN,
    MARKDOWN_HR_PATTERN,
    MARKDOWN_FOOTNOTE_PATTERN,
    MARKDOWN_DEFINITION_PATTERN,
    MARKDOWN_TASK_PATTERN,
    MARKDOWN_AUTOLINK_PATTERN,
    MARKDOWN_REFERENCE_PATTERN,
    MARKDOWN_COMMENT_PATTERN,
    MARKDOWN_METADATA_PATTERN,
    MARKDOWN_EXTENSION_PATTERN,
    PYTHON_PATTERN,
    JAVASCRIPT_PATTERN,
    JAVA_PATTERN,
    CPP_PATTERN,
    CSHARP_PATTERN,
    GO_PATTERN,
    RUST_PATTERN,
    PHP_PATTERN,
    RUBY_PATTERN,
    SWIFT_PATTERN,
    KOTLIN_PATTERN,
    DART_PATTERN,
    TYPESCRIPT_PATTERN,
    SQL_PATTERN,
    SHELL_PATTERN,
    POWERSHELL_PATTERN,
    REGEX_PATTERN,
    COMMENT_PATTERN,
    HTML_TAG_PATTERN,
    XML_TAG_PATTERN,
    CSS_SELECTOR_PATTERN,
    MATH_PATTERN,
    JSON_PATTERN,
    YAML_PATTERN,
    INI_PATTERN,
    CONFIG_PATTERN,
    LOG_PATTERN,
    TIMESTAMP_PATTERN,
    UUID_PATTERN,
    MAC_PATTERN,
    IP_PATTERN,
    IPV6_PATTERN,
    URL_PATH_PATTERN,
    FILE_EXT_PATTERN,
    CLI_PATTERN,
    ENV_PATTERN,
    EMOJI_TEXT_PATTERN,
    PUNCTUATION_PATTERN,
    FORMAT_PATTERN,
)


class CharacterNameValidator:
    """"""

    def __init__(self):
        self.non_character_keywords = set(NON_CHARACTER_KEYWORDS)
        self.character_type_keywords = set(CHARACTER_TYPE_KEYWORDS)
        self.allowed_symbols = set(ALLOWED_SYMBOLS)

    def is_character_name(self, text: str) -> bool:
        """
        

        Args:
            text: 

        Returns:
            bool: 
        """
        if not text:
            return False

        text_stripped = text.strip()

        # 
        if not self._check_length(text_stripped):
            return False

        # 
        if not self._check_keywords(text_stripped):
            return False

        # 
        if not self._check_character_types(text_stripped):
            return False

        # 
        if not self._check_formats(text_stripped):
            return False

        # 
        if not self._check_code_patterns(text_stripped):
            return False

        # Markdown
        if not self._check_markdown_patterns(text_stripped):
            return False

        # 
        if not self._check_special_chars(text_stripped):
            return False

        return True

    def _check_length(self, text: str) -> bool:
        """"""
        if len(text) < LENGTH_MIN or len(text) > LENGTH_MAX:
            return False
        if text.count(" ") > MAX_SPACES:
            return False
        if text.count("-") > MAX_HYPHENS:
            return False
        if text.count(".") > MAX_DOTS:
            return False
        return True

    def _check_keywords(self, text: str) -> bool:
        """"""
        for keyword in self.non_character_keywords:
            if keyword in text:
                return False
        return True

    def _check_character_types(self, text: str) -> bool:
        """"""
        # 
        if text.isdigit():
            return False

        # 
        if not any(c.isalnum() for c in text):
            return False

        # 
        has_chinese = bool(re.search(CHINESE_PATTERN, text))

        # 
        if re.search(FULL_WIDTH_ALPHA_NUM_PATTERN, text):
            return False

        # 
        if not has_chinese and re.search(HALF_WIDTH_ALPHA_NUM_PATTERN, text):
            return False

        # 
        if any(c.isupper() for c in text) and any(c.islower() for c in text):
            return False

        return True

    def _check_formats(self, text: str) -> bool:
        """"""
        patterns = [
            (VERSION_PATTERN, ""),
            (DATE_PATTERN, ""),
            (TIME_PATTERN, ""),
            (URL_PATTERN, "URL"),
            (EMAIL_PATTERN, ""),
            (PATH_PATTERN, ""),
            (TIMESTAMP_PATTERN, ""),
            (UUID_PATTERN, "UUID"),
            (MAC_PATTERN, "MAC"),
            (IP_PATTERN, "IP"),
            (IPV6_PATTERN, "IPv6"),
            (URL_PATH_PATTERN, "URL"),
            (FILE_EXT_PATTERN, ""),
        ]

        for pattern, _ in patterns:
            if re.search(pattern, text):
                return False
        return True

    def _check_code_patterns(self, text: str) -> bool:
        """"""
        code_patterns = [
            COMMAND_PATTERN,
            CODE_PATTERN,
            TAG_PATTERN,
            BRACKET_PATTERN,
            PYTHON_PATTERN,
            JAVASCRIPT_PATTERN,
            JAVA_PATTERN,
            CPP_PATTERN,
            CSHARP_PATTERN,
            GO_PATTERN,
            RUST_PATTERN,
            PHP_PATTERN,
            RUBY_PATTERN,
            SWIFT_PATTERN,
            KOTLIN_PATTERN,
            DART_PATTERN,
            TYPESCRIPT_PATTERN,
            SQL_PATTERN,
            SHELL_PATTERN,
            POWERSHELL_PATTERN,
            REGEX_PATTERN,
            COMMENT_PATTERN,
            HTML_TAG_PATTERN,
            XML_TAG_PATTERN,
            CSS_SELECTOR_PATTERN,
            MATH_PATTERN,
            JSON_PATTERN,
            YAML_PATTERN,
            INI_PATTERN,
            CONFIG_PATTERN,
            LOG_PATTERN,
            CLI_PATTERN,
            ENV_PATTERN,
        ]

        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True

    def _check_markdown_patterns(self, text: str) -> bool:
        """Markdown"""
        md_patterns = [
            MARKDOWN_PATTERN,
            MARKDOWN_LINK_PATTERN,
            MARKDOWN_IMAGE_PATTERN,
            MARKDOWN_HEADING_PATTERN,
            MARKDOWN_LIST_PATTERN,
            MARKDOWN_QUOTE_PATTERN,
            MARKDOWN_CODE_PATTERN,
            MARKDOWN_BOLD_PATTERN,
            MARKDOWN_ITALIC_PATTERN,
            MARKDOWN_STRIKETHROUGH_PATTERN,
            MARKDOWN_INLINE_CODE_PATTERN,
            MARKDOWN_TABLE_PATTERN,
            MARKDOWN_HR_PATTERN,
            MARKDOWN_FOOTNOTE_PATTERN,
            MARKDOWN_DEFINITION_PATTERN,
            MARKDOWN_TASK_PATTERN,
            MARKDOWN_AUTOLINK_PATTERN,
            MARKDOWN_REFERENCE_PATTERN,
            MARKDOWN_COMMENT_PATTERN,
            MARKDOWN_METADATA_PATTERN,
            MARKDOWN_EXTENSION_PATTERN,
        ]

        for pattern in md_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True

    def _check_special_chars(self, text: str) -> bool:
        """"""
        # 
        for char in text:
            if not char.isalnum() and char not in self.allowed_symbols:
                return False

        # emoji
        if re.search(EMOJI_PATTERN, text):
            return False

        # emoji
        if re.search(EMOJI_TEXT_PATTERN, text):
            return False
