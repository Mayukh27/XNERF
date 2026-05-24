from __future__ import annotations

from xnerf.preprocessing.disassembler import DisassemblerProcessor
from xnerf.preprocessing.isr_builder import ISRBuilderProcessor
from xnerf.preprocessing.semantic_mapper import SemanticMapperProcessor
from xnerf.utils.base import Processor


class ArchitectureNormalizationPipeline(Processor):
    """End-to-end bytes -> ISR tensor pipeline.

    Inputs:
        {"bytes": bytes, "arch": str}
    Outputs:
        torch.LongTensor [T, 4]
    Tensor dimensions:
        [max_len, 4]
    Usage:
        isr = ArchitectureNormalizationPipeline("arm64").process({"bytes": blob, "arch": "arm64"})
    """

    def __init__(
        self,
        arch: str = "x86",
        disassembler: DisassemblerProcessor | None = None,
        mapper: SemanticMapperProcessor | None = None,
        builder: ISRBuilderProcessor | None = None,
    ):
        self.disassembler = disassembler or DisassemblerProcessor(arch=arch)
        self.mapper = mapper or SemanticMapperProcessor()
        self.builder = builder or ISRBuilderProcessor()

    def process(self, item):
        instructions = self.disassembler.process(item)
        mapped = self.mapper.process(instructions)
        arch = item.get("arch", "x86") if isinstance(item, dict) else "x86"
        return self.builder.process({"semantic": mapped, "arch": arch})

