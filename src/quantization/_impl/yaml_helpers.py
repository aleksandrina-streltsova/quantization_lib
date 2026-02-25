from pathlib import PosixPath, Path

import yaml

from quantization.aggregation import AggregationPlan, Aggregation, AggregationRegistry
from quantization.config import QuantizationConfig
from quantization.quant_columns import RoundingQuantColumnConfig, FixedEdgesQuantColumnConfig, \
    FixedStepQuantColumnConfig


def _add_mapping_representer(cls, attrs: list[str], flow_style: bool = False):
    def _representer(dumper: yaml.Dumper, data):
        mapping = {a: getattr(data, a) for a in attrs if getattr(data, a) is not None}
        return dumper.represent_mapping(cls.__name__, mapping, flow_style=flow_style)

    yaml.add_representer(cls, _representer)


def _add_mapping_constructor(cls, loader):
    def _constructor(loader: yaml.Loader, node: yaml.Node):
        values = loader.construct_mapping(node)
        return cls(**values)

    yaml.add_constructor(cls.__name__, _constructor, loader)


def yaml_add_representers():
    yaml.add_representer(PosixPath, lambda dumper, data: dumper.represent_scalar("Path", str(data)))

    _add_mapping_representer(FixedStepQuantColumnConfig, ["name", "step", "clip"], flow_style=True)
    _add_mapping_representer(FixedEdgesQuantColumnConfig, ["name", "edges", "clip"], flow_style=True)
    _add_mapping_representer(RoundingQuantColumnConfig, ["name", "decimals", "clip"], flow_style=True)

    yaml.add_multi_representer(Aggregation, lambda dumper, data: dumper.represent_str(data.name))

    _add_mapping_representer(AggregationPlan, ["per_column", "default_non_quant", "default_quant"])
    _add_mapping_representer(QuantizationConfig, ["quant_columns", "aggregations", "output_dir"])


def yaml_add_constructors(loader, agg_registry: AggregationRegistry):
    yaml.add_constructor(str(Path.__name__), lambda loader, node: Path(loader.construct_scalar(node)), loader)

    _add_mapping_constructor(FixedStepQuantColumnConfig, loader)
    _add_mapping_constructor(FixedEdgesQuantColumnConfig, loader)
    _add_mapping_constructor(RoundingQuantColumnConfig, loader)

    yaml.add_constructor(
        Aggregation.__name__,
        lambda loader, node: agg_registry.get(loader.construct_scalar(node)),
        loader,
    )

    _add_mapping_constructor(AggregationPlan, loader)
    _add_mapping_constructor(QuantizationConfig, loader)
