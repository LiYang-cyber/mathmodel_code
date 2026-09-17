"""Task-neutral workflow composition for modeling modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

StepFunction = Callable[[dict[str, Any]], Mapping[str, Any] | None]


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    function: StepFunction
    requires: frozenset[str] = field(default_factory=frozenset)
    provides: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.name:
            raise ValueError("步骤名称不能为空")


class Workflow:
    """Execute validated steps over a shared context.

    A step receives a shallow copy of the current context and returns only its updates.
    This keeps preprocessing, modeling, validation and reporting functions independently reusable.
    """

    def __init__(self, steps: Iterable[WorkflowStep]):
        self.steps = list(steps)
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("流程步骤名称必须唯一")

    def validate(self, available: Iterable[str] = ()) -> frozenset[str]:
        keys = set(available)
        for step in self.steps:
            missing = step.requires - keys
            if missing:
                raise ValueError(f"步骤 {step.name!r} 缺少输入: {', '.join(sorted(missing))}")
            keys.update(step.provides)
        return frozenset(keys)

    def run(self, initial: Mapping[str, Any] | None = None) -> dict[str, Any]:
        context = dict(initial or {})
        self.validate(context)
        for step in self.steps:
            updates = step.function(dict(context))
            if updates is None:
                updates = {}
            if not isinstance(updates, Mapping):
                raise TypeError(f"步骤 {step.name!r} 必须返回 Mapping 或 None")
            undeclared = set(updates) - step.provides
            if undeclared:
                raise ValueError(f"步骤 {step.name!r} 返回未声明字段: {', '.join(sorted(undeclared))}")
            missing = step.provides - set(updates)
            if missing:
                raise ValueError(f"步骤 {step.name!r} 未提供声明字段: {', '.join(sorted(missing))}")
            context.update(updates)
        return context


class ComponentRegistry:
    """Register factories for models, transforms, validators and reporters."""

    def __init__(self):
        self._components: dict[str, dict[str, Callable[..., Any]]] = {}

    def register(self, category: str, name: str, factory: Callable[..., Any]) -> None:
        if not category or not name or not callable(factory):
            raise ValueError("category、name 必须非空且 factory 必须可调用")
        bucket = self._components.setdefault(category, {})
        if name in bucket:
            raise ValueError(f"组件已存在: {category}/{name}")
        bucket[name] = factory

    def create(self, category: str, name: str, **params):
        try:
            factory = self._components[category][name]
        except KeyError as exc:
            choices = ", ".join(self.names(category)) or "无"
            raise KeyError(f"未知组件 {category}/{name}；可选值: {choices}") from exc
        return factory(**params)

    def names(self, category: str) -> tuple[str, ...]:
        return tuple(sorted(self._components.get(category, {})))


def default_registry() -> ComponentRegistry:
    """Build a registry spanning tabular, clustering, anomaly and optional deep models."""
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    registry = ComponentRegistry()
    components = {
        "transform": {
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "zscore": StandardScaler,
        },
        "classification": {
            "logistic": LogisticRegression,
            "random_forest": RandomForestClassifier,
        },
        "regression": {
            "linear": LinearRegression,
            "random_forest": RandomForestRegressor,
        },
        "clustering": {"dbscan": DBSCAN, "kmeans": KMeans},
        "anomaly": {"isolation_forest": IsolationForest},
    }
    for category, factories in components.items():
        for name, factory in factories.items():
            registry.register(category, name, factory)
    return registry
