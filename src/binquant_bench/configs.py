import itertools
from dataclasses import dataclass

from qdrant_client import models

OVERSAMPLING = [1.0, 2.0, 3.0, 4.0]
BITS = models.TurboQuantBitSize
TURBO_COMPRESSION = {BITS.BITS4: "8x", BITS.BITS2: "16x", BITS.BITS1_5: "~21x", BITS.BITS1: "32x"}


@dataclass(frozen=True)
class SearchVariant:
    rescore: bool | None = None
    oversampling: float | None = None

    def params(self) -> models.SearchParams | None:
        if self.rescore is None:
            return None
        return models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=self.rescore, oversampling=self.oversampling
            )
        )


@dataclass(frozen=True)
class Config:
    name: str
    notes: str
    datatype: models.Datatype = models.Datatype.FLOAT32
    quantization: models.QuantizationConfig | None = None
    sweep: bool = True

    @property
    def slug(self) -> str:
        return self.name.replace("+", "-")

    @property
    def variants(self) -> list[SearchVariant]:
        if self.quantization is None or not self.sweep:
            return [SearchVariant()]
        grid = itertools.product([False, True], OVERSAMPLING)
        return [SearchVariant(rescore, oversampling) for rescore, oversampling in grid]


def binary(
    encoding: models.BinaryQuantizationEncoding | None = None,
    query_encoding: models.BinaryQuantizationQueryEncoding | None = None,
) -> models.BinaryQuantization:
    return models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            encoding=encoding, query_encoding=query_encoding, memory=models.Memory.PINNED
        )
    )


def turbo(bits: models.TurboQuantBitSize) -> models.TurboQuantization:
    return models.TurboQuantization(
        turbo=models.TurboQuantQuantizationConfig(bits=bits, memory=models.Memory.PINNED)
    )


def scalar_int8() -> models.ScalarQuantization:
    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, memory=models.Memory.PINNED
        )
    )


def build_configs(is_120: bool = False) -> list[Config]:
    encodings = models.BinaryQuantizationEncoding
    query_encodings = models.BinaryQuantizationQueryEncoding
    configs = [
        Config("float32", "baseline, HNSW on full-precision vectors"),
        Config("scalar-int8", "4x", quantization=scalar_int8()),
        Config("binary-1bit", "32x", quantization=binary()),
        Config("binary-2bit", "16x", quantization=binary(encodings.TWO_BITS)),
        Config("binary-1_5bit", "~21x", quantization=binary(encodings.ONE_AND_HALF_BITS)),
        Config(
            "binary-1bit+query-scalar8",
            "32x, query encoded as 8-bit scalar (asymmetric)",
            quantization=binary(query_encoding=query_encodings.SCALAR8BITS),
        ),
        *(
            Config(f"turbo-{bits.value}", note, quantization=turbo(bits))
            for bits, note in TURBO_COMPRESSION.items()
        ),
        Config(
            "turbo4-datatype",
            "turbo4 as the only stored representation, no rescoring possible",
            datatype=models.Datatype.TURBO4,
            sweep=False,
        ),
        Config(
            "turbo4-datatype+turbo-bits1",
            "1-bit TurboQuant index, rescored against turbo4 originals",
            datatype=models.Datatype.TURBO4,
            quantization=turbo(BITS.BITS1),
        ),
    ]

    if is_120:
        configs.extend(
            [
                Config(
                    "turbo8-datatype",
                    "turbo8 is a 8-bits datatype derived from TurboQuant",
                    datatype=models.Datatype.TURBO8,
                    sweep=False,
                ),
                Config(
                    "turbo16-datatype",
                    "turbo16 is a 16-bits datatype derived from TurboQuant",
                    datatype=models.Datatype.TURBO16,
                    sweep=False,
                ),
                Config(
                    "turbo8-datatype+turbo-bits2",
                    "turbo8 is a 8-bits datatype derived from TurboQuant",
                    datatype=models.Datatype.TURBO8,
                    quantization=turbo(bits=BITS.BITS2),
                ),
                Config(
                    "turbo16-datatype+turbo-bits4",
                    "turbo16 is a 16-bits datatype derived from TurboQuant",
                    datatype=models.Datatype.TURBO16,
                    quantization=turbo(bits=BITS.BITS4),
                ),
            ]
        )

    return configs
