import os
import time
import random
import json
from google import genai
from mistralai import Mistral
from config import (
    FLASH_VLM_SETTINGS,
    SOTA_VLM_SETTINGS,
    LLM_SETTINGS,
    VLM_SETTINGS_MIS,
    LLM_SETTINGS_MIS,
)
from vlm_utils.output_structure import Instance, KinematicRelationship
import vlm_utils.gemini_message as gemini_message
import vlm_utils.pixtral_message as pixtral_message


class BaseVLMClient:
    """
    Abstract base class defining VLM client interface.
    """

    def __init__(self):
        self.provider = None
        raise NotImplementedError

    def infer(
        self, msg, response_format=None, model_index=0
    ):  # model index 0 for llm, 1 for vlm, 2 for sota vlm
        raise NotImplementedError


class GeminiVLMClient(BaseVLMClient):
    def __init__(self):
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            raise RuntimeError("GENAI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        self.flash_vlm = FLASH_VLM_SETTINGS["model_name"]
        self.flash_vlm_max_tokens = FLASH_VLM_SETTINGS["max_tokens"]
        self.flash_vlm_temperature = FLASH_VLM_SETTINGS["temperature"]
        self.sota_vlm = SOTA_VLM_SETTINGS["model_name"]
        self.sota_vlm_max_tokens = SOTA_VLM_SETTINGS["max_tokens"]
        self.sota_vlm_temperature = SOTA_VLM_SETTINGS["temperature"]
        self.llm = LLM_SETTINGS["model_name"]
        self.llm_max_tokens = LLM_SETTINGS["max_tokens"]
        self.llm_temperature = LLM_SETTINGS["temperature"]
        self.provider = "GEMINI"

    def infer(self, msg, response_format=None, model_index=0):
        max_retries = 15
        base_delay = 2  # Base delay in seconds

        attempt = 0
        chat_response = {"text": None}
        while attempt < max_retries:
            try:
                if response_format == None:
                    chat_response = self.client.models.generate_content(
                        model=self.flash_vlm if model_index <= 1 else self.sota_vlm,
                        contents=msg,
                    )
                    print(chat_response)
                    return json.loads(chat_response.text)
                else:
                    print("using format")
                    chat_response = self.client.models.generate_content(
                        model=self.flash_vlm if model_index <= 1 else self.sota_vlm,
                        contents=msg,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": response_format,
                        },
                    )
                    print(chat_response)
                    return json.loads(chat_response.text)

            except Exception as e:
                # Check if it's a rate limit error
                if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                    or "overloaded" in str(e).lower()
                    or "disconnected" in str(e).lower()
                ):
                    attempt += 1
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        # Calculate exponential backoff with jitter
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        print(
                            f"Rate limit exceeded. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"Failed after {max_retries} attempts due to rate limiting."
                        )
                        raise
                else:
                    print(f"API error: {e}")
                    attempt += 1
                    # If it's not a rate limit error, retry as well
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        print(
                            f"Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise


class MistralVLMClient(BaseVLMClient):
    def __init__(self):
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable not set")
        self.client = Mistral(api_key=api_key)

        # vision–language settings
        self.vlm = VLM_SETTINGS_MIS["model_name"]
        self.vlm_max_tokens = VLM_SETTINGS_MIS["max_tokens"]
        self.vlm_temperature = VLM_SETTINGS_MIS["temperature"]
        # pure-text LLM settings
        self.llm = LLM_SETTINGS_MIS["model_name"]
        self.llm_max_tokens = LLM_SETTINGS_MIS["max_tokens"]
        self.llm_temperature = LLM_SETTINGS_MIS["temperature"]
        self.provider = "MISTRAL"

    def infer(self, msg, response_format=None, model_index=0):
        """
        msg may now be:
          - a tuple (messages_list, vis_img) from pixtral_message.part_relation_msg_for_KAF
          - a plain list of {"role","content"} dicts
          - a bare string (we’ll wrap it)
        """
        if isinstance(msg, tuple) and isinstance(msg[0], list):
            messages = msg[0]
        elif isinstance(msg, list) and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in msg
        ):
            messages = msg
        else:
            messages = [{"role": "user", "content": msg}]

        if model_index == 0:
            model_name = self.llm
            max_tokens = self.llm_max_tokens
            temperature = self.llm_temperature
            tag = "LLM"
        else:
            model_name = self.vlm
            max_tokens = self.vlm_max_tokens
            temperature = self.vlm_temperature
            tag = "VLM"

        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                if response_format is None:
                    resp = self.client.chat.complete(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return {"response": resp.choices[0].message.content}
                else:
                    resp = self.client.chat.parse(
                        model=model_name,
                        messages=messages,
                        response_format=response_format,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return json.loads(resp.choices[0].message.content)

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.random()
                    print(
                        f"[{tag}] rate-limit; retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    print(f"[{tag}] API error on attempt {attempt + 1}: {e}")
                    raise


def create_vlm_client(provider: str) -> BaseVLMClient:
    """
    Factory for creating VLM client based on provider.
    """
    provider = provider.upper()
    if provider == "GEMINI":
        return GeminiVLMClient()
    elif provider == "MISTRAL":
        return MistralVLMClient()
    else:
        raise ValueError(f"Unsupported VLM provider: {provider}")


class VLMService:
    def __init__(self, provider: str):
        provider_upper = provider.upper()
        self.client = create_vlm_client(provider_upper)
        if provider_upper == "MISTRAL":
            self.msg_mod = pixtral_message
        elif provider_upper == "GEMINI":
            self.msg_mod = gemini_message
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def instance_description(
        self, src_image_dir, image_id, p_mask_dir, bbox, debug=False
    ):
        # Build the VLM prompt
        msg = self.msg_mod.instance_description_msg(
            os.path.join(src_image_dir, f"{image_id}.png"),
            p_mask_dir,
            crop_config=self.msg_mod.crop_config(
                True,
                bbox,
                padding_box=[-20, -20, 20, 20],
            ),
            debug=debug,
        )
        instance_desc = {"valid": False}
        if self.client.provider == "GEMINI":
            # Parse directly into the Instance schema via SOTA VLM
            instance_desc = self.client.infer(msg, Instance, model_index=2)
        elif self.client.provider == "MISTRAL":
            # Generate description via VLM
            instance_desc = self.client.infer(msg, None, model_index=1)
            # Parse into Instance via LLM
            msg = self.msg_mod.parse_instance_description_msg(instance_desc["response"])
            instance_desc = self.client.infer(msg, Instance, model_index=0)

        # Normalize valid → boolean
        print(instance_desc)
        if instance_desc["valid"] == "Yes":
            instance_desc["valid"] = True
        else:
            return {"valid": False}
        return instance_desc

    def kinematic_description(self, src_img_path, pair, name, bbox, debug=False):
        # part_relation_msg_for_KAF returns (messages, vis_img)
        msg_vis = self.msg_mod.part_relation_msg_for_KAF(
            src_img_path,
            pair[0],
            pair[1],
            name,
            crop_config=self.msg_mod.crop_config(
                True,
                bbox,
                padding_box=[-20, -20, 20, 20],
            ),
            debug=debug,
        )
        # Unpack prompt messages vs. visualization
        if isinstance(msg_vis, tuple):
            prompt_msg, vis_img = msg_vis
        else:
            prompt_msg, vis_img = msg_vis, None

        # Parse into KinematicRelationship via FLASH VLM
        kinematic_desc = self.client.infer(
            prompt_msg, KinematicRelationship, model_index=1
        )

        if not debug:
            vis_img = None
        return kinematic_desc, vis_img
