from typing import Literal

from pydantic import BaseModel, model_validator


class Action(BaseModel):
    """Single step the browser agent takes.

    Action-dependent validation:
    - click: target required
    - type:  target + value required
    - scroll / scroll_up / done: target optional
    """

    action: Literal["click", "type", "scroll", "scroll_up", "done"]
    target: str = ""
    value: str | None = None
    reason: str

    @model_validator(mode="after")
    def validate_fields(self) -> "Action":
        self.reason = self.reason.strip()
        if not self.reason:
            raise ValueError("reason must not be empty")

        self.target = self.target.strip()

        if self.action == "click" and not self.target:
            raise ValueError("target required for click actions")

        if self.action == "type":
            if not self.target:
                raise ValueError("target required for type actions")
            if not self.value:
                raise ValueError("value required for type actions")

        return self
