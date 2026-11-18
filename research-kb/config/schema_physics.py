"""Pydantic schemas for structured physics-parameter extraction (Layer 2).

Design notes
------------
* Every extracted number carries an ``evidence_quote`` copied verbatim from the
  paper; the extractor rejects any record whose quote cannot be located in the
  source text (anti-hallucination check).
* Values are stored as reported *and* normalised to a canonical unit per
  quantity so cross-paper comparison is a plain SQL query.
* The relational layout (paper -> material -> measurement(method)) is already a
  Material--Property--Method--Paper graph; see ``kg_edges`` view in the DB.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Quantity(str, Enum):
    curie_temperature = "T_C"
    neel_temperature = "T_N"
    blocking_temperature = "T_B"
    dmi_interfacial = "D_int"          # interfacial / bulk DMI constant (energy per area)
    dmi_bulk = "D_bulk"                # bulk DMI (energy per volume / length) when given as such
    saturation_magnetization = "M_s"
    exchange_stiffness = "A_ex"
    uniaxial_anisotropy = "K_u"
    effective_anisotropy = "K_eff"
    anisotropy_field = "H_K"
    coercive_field = "H_c"
    gilbert_damping = "alpha"
    skyrmion_diameter = "d_sk"
    domain_wall_width = "delta_DW"
    domain_period = "lambda_domain"
    spin_hall_angle = "theta_SH"
    spin_polarization = "P_spin"
    exchange_bias = "H_EB"
    magnetic_moment = "m_atom"
    critical_current_density = "J_c"
    domain_wall_velocity = "v_DW"
    skyrmion_hall_angle = "theta_SkH"
    anomalous_hall_conductivity = "sigma_AH"
    topological_hall_resistivity = "rho_THE"
    thickness_critical = "t_c"
    # magnonics
    spin_wave_group_velocity = "v_g"
    spin_wave_decay_length = "L_att"
    spin_wave_frequency = "f_SW"
    spin_wave_wavelength = "lambda_SW"
    fmr_linewidth = "dH_FMR"
    magnon_lifetime = "tau_magnon"
    # MTJ / memory
    tmr_ratio = "TMR"
    ra_product = "RA"
    thermal_stability = "Delta"
    switching_current_density = "J_sw"
    switching_time = "t_sw"
    write_energy = "E_write"
    # spin-torque oscillators
    sto_frequency = "f_STO"
    sto_linewidth = "df_STO"
    sto_output_power = "P_STO"
    # neuromorphic / reservoir computing
    task_accuracy = "accuracy"
    memory_capacity = "MC"
    energy_per_operation = "E_op"
    other = "other"


# canonical unit per quantity; conversions below map common reported units to it
CANONICAL_UNIT: dict[Quantity, str] = {
    Quantity.curie_temperature: "K", Quantity.neel_temperature: "K",
    Quantity.blocking_temperature: "K",
    Quantity.dmi_interfacial: "mJ/m^2", Quantity.dmi_bulk: "mJ/m^2",
    Quantity.saturation_magnetization: "kA/m",
    Quantity.exchange_stiffness: "pJ/m",
    Quantity.uniaxial_anisotropy: "MJ/m^3", Quantity.effective_anisotropy: "MJ/m^3",
    Quantity.anisotropy_field: "T", Quantity.coercive_field: "T", Quantity.exchange_bias: "T",
    Quantity.gilbert_damping: "", Quantity.spin_hall_angle: "", Quantity.spin_polarization: "",
    Quantity.skyrmion_diameter: "nm", Quantity.domain_wall_width: "nm",
    Quantity.domain_period: "nm", Quantity.thickness_critical: "nm",
    Quantity.magnetic_moment: "mu_B",
    Quantity.critical_current_density: "A/m^2",
    Quantity.domain_wall_velocity: "m/s",
    Quantity.skyrmion_hall_angle: "deg",
    Quantity.anomalous_hall_conductivity: "S/cm",
    Quantity.topological_hall_resistivity: "uOhm cm",
    Quantity.spin_wave_group_velocity: "m/s", Quantity.spin_wave_decay_length: "nm",
    Quantity.spin_wave_frequency: "GHz", Quantity.spin_wave_wavelength: "nm",
    Quantity.fmr_linewidth: "T", Quantity.magnon_lifetime: "ns",
    Quantity.tmr_ratio: "", Quantity.ra_product: "Ohm um^2", Quantity.thermal_stability: "",
    Quantity.switching_current_density: "A/m^2", Quantity.switching_time: "ns",
    Quantity.write_energy: "fJ", Quantity.energy_per_operation: "fJ",
    Quantity.sto_frequency: "GHz", Quantity.sto_linewidth: "GHz", Quantity.sto_output_power: "nW",
    Quantity.task_accuracy: "", Quantity.memory_capacity: "",
}

_U = lambda s: re.sub(r"[\s·*]", "", s.replace("μ", "u").replace("µ", "u").replace("Ω", "Ohm")
                       .replace("²", "^2").replace("³", "^3").replace("−", "-")).lower()

# (canonical unit) -> {normalised reported unit: factor}
_CONV: dict[str, dict[str, float]] = {
    "K": {"k": 1.0},
    "mJ/m^2": {"mj/m^2": 1.0, "mjm^-2": 1.0, "mj/m2": 1.0, "j/m^2": 1e3, "pj/m^2": 1e-9,
               "erg/cm^2": 1.0, "uj/m^2": 1e-3},
    "kA/m": {"ka/m": 1.0, "a/m": 1e-3, "ma/m": 1e3, "emu/cm^3": 1.0, "emu/cc": 1.0,
             "kam^-1": 1.0},
    "pJ/m": {"pj/m": 1.0, "j/m": 1e12, "erg/cm": 1e7, "pjm^-1": 1.0},
    "MJ/m^3": {"mj/m^3": 1.0, "j/m^3": 1e-6, "kj/m^3": 1e-3, "erg/cm^3": 1e-7, "erg/cc": 1e-7,
               "mjm^-3": 1.0},
    "T": {"t": 1.0, "mt": 1e-3, "oe": 1e-4, "koe": 0.1, "g": 1e-4},
    "nm": {"nm": 1.0, "um": 1e3, "a": 0.1, "å": 0.1, "pm": 1e-3},
    "A/m^2": {"a/m^2": 1.0, "a/cm^2": 1e4, "ma/cm^2": 1e10},
    "m/s": {"m/s": 1.0, "ms^-1": 1.0, "km/s": 1e3},
    "S/cm": {"s/cm": 1.0, "ohm^-1cm^-1": 1.0, "s/m": 1e-2},
    "uOhm cm": {"uohmcm": 1.0, "nohmcm": 1e-3, "mohmcm": 1e3, "ohmcm": 1e6},
    "mu_B": {"mu_b": 1.0, "ub": 1.0, "μb": 1.0, "mub": 1.0, "u_b": 1.0},
    "deg": {"deg": 1.0, "°": 1.0, "degree": 1.0, "degrees": 1.0},
    "": {"": 1.0, "%": 0.01},
    "GHz": {"ghz": 1.0, "mhz": 1e-3, "hz": 1e-9, "thz": 1e3},
    "ns": {"ns": 1.0, "ps": 1e-3, "us": 1e3, "ms": 1e6, "s": 1e9},
    "fJ": {"fj": 1.0, "aj": 1e-3, "pj": 1e3, "nj": 1e6, "j": 1e15},
    "nW": {"nw": 1.0, "pw": 1e-3, "uw": 1e3, "mw": 1e6, "w": 1e9},
    "Ohm um^2": {"ohmum^2": 1.0, "ohmμm^2": 1.0, "ohm-um^2": 1.0, "ohmcm^2": 1e8, "kohmum^2": 1e3,
                 "mohmum^2": 1e-3},
}


def normalize_value(quantity: Quantity, value: float | None, unit: str | None
                    ) -> tuple[float | None, str | None]:
    """Convert ``value [unit]`` into the quantity's canonical unit. Returns (None, None) if unknown."""
    if value is None:
        return None, None
    canon = CANONICAL_UNIT.get(quantity)
    if canon is None:
        return None, None
    table = _CONV.get(canon, {})
    u = _U(unit or "")
    if quantity == Quantity.curie_temperature and u in ("°c", "c"):
        return value + 273.15, canon
    if u in table:
        return value * table[u], canon
    return None, None


_LATEX_SUB = re.compile(r"_\{?(\d+(?:\.\d+)?|[xyzδ\-+])\}?")


def normalize_formula(formula: str) -> str:
    """'Fe$_3$GaTe$_2$' / 'Fe_{3}GaTe_{2}' / r'\\text{Fe}_3\\text{GaTe}_2' -> 'Fe3GaTe2'."""
    s = formula or ""
    s = re.sub(r"\\(?:text|mathrm|rm|textrm)\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "")
    s = _LATEX_SUB.sub(r"\1", s)
    s = re.sub(r"[{}\\\s]", "", s)
    return s


def composition_key(formula: str) -> str:
    """Order-independent key (Hill-like): 'GaFe3Te2' and 'Fe3GaTe2' -> 'Fe3Ga1Te2'."""
    f = normalize_formula(formula)
    parts = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", f)
    if not parts:
        return f.lower()
    comp: dict[str, float] = {}
    for el, n in parts:
        comp[el] = comp.get(el, 0.0) + (float(n) if n else 1.0)
    return "".join(f"{el}{comp[el]:g}" for el in sorted(comp))


class SampleForm(str, Enum):
    bulk_single_crystal = "bulk_single_crystal"
    polycrystal = "polycrystal"
    exfoliated_flake = "exfoliated_flake"
    thin_film = "thin_film"
    heterostructure = "heterostructure"
    monolayer = "monolayer"
    nanostructure = "nanostructure"
    simulation = "simulation"
    theory_dft = "theory_dft"
    unknown = "unknown"


class Method(str, Enum):
    moke = "MOKE"
    polar_moke = "polar_MOKE"
    kerr_microscopy = "Kerr_microscopy"
    ltem = "LTEM"
    mfm = "MFM"
    xmcd = "XMCD"
    xmcd_peem = "XMCD-PEEM"
    stxm = "STXM"
    spleem = "SPLEEM"
    nv_magnetometry = "NV_magnetometry"
    sans = "SANS"
    neutron = "neutron_diffraction"
    bls = "BLS"
    fmr = "FMR"
    st_fmr = "ST-FMR"
    squid = "SQUID"
    vsm = "VSM"
    hall_transport = "Hall_transport"
    harmonic_hall = "harmonic_Hall"
    domain_wall_motion = "domain_wall_motion"
    domain_width_analysis = "domain_width_analysis"
    asymmetric_bubble_expansion = "asymmetric_bubble_expansion"
    dft = "DFT"
    micromagnetic = "micromagnetic_simulation"
    monte_carlo = "Monte_Carlo"
    atomistic = "atomistic_spin_dynamics"
    tr_moke = "TR-MOKE"
    micro_bls = "micro-BLS"
    vna_fmr = "VNA-FMR"
    psws = "propagating_spin_wave_spectroscopy"
    tmr_measurement = "TMR_transport"
    spectrum_analyzer = "spectrum_analyzer"
    benchmark_task = "computing_benchmark_task"
    other = "other"


class MaterialSystem(BaseModel):
    formula: str = Field(..., description="Chemical formula as written, e.g. 'Fe3GaTe2' or 'Pt/Co/Ta'")
    stack: Optional[str] = Field(None, description="Full heterostructure stack incl. thicknesses, e.g. 'Ta(5)/Pt(3)/Co(0.8)/MgO(2)'")
    doping: Optional[str] = None
    sample_form: SampleForm = SampleForm.unknown
    thickness_nm: Optional[float] = Field(None, description="Magnetic layer / flake thickness in nm")
    layers: Optional[int] = Field(None, description="Number of vdW layers if stated")
    substrate: Optional[str] = None


class ParameterRecord(BaseModel):
    material: MaterialSystem
    quantity: Quantity
    quantity_label: str = Field(..., description="Symbol/name as the paper writes it, e.g. 'D', '|D_s|', 'T_C'")
    value: Optional[float] = Field(None, description="Central reported value (float). Null if only a bound/range")
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    uncertainty: Optional[float] = None
    unit: str = Field("", description="Unit as reported, e.g. 'mJ/m^2', 'K', 'emu/cm^3'")
    temperature_K: Optional[float] = Field(None, description="Measurement temperature in K, if stated")
    field_T: Optional[float] = Field(None, description="Applied field in T, if stated")
    method: Method = Method.other
    method_detail: Optional[str] = None
    is_theoretical: bool = False
    evidence_quote: str = Field(..., description="VERBATIM sentence/table cell from the text supporting the value (<= 300 chars)")
    section: Optional[str] = None
    confidence: float = Field(0.7, ge=0.0, le=1.0)

    @field_validator("evidence_quote")
    @classmethod
    def _strip_quote(cls, v: str) -> str:
        return v.strip()[:400]


class PaperExtraction(BaseModel):
    """Whole-paper extraction result returned by the LLM (strict JSON)."""
    materials_studied: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list, description="<= 5 one-sentence findings")
    parameters: list[ParameterRecord] = Field(default_factory=list)


def json_schema_hint() -> str:
    """Compact schema description inserted into the extraction prompt."""
    q = ", ".join(f"{x.value}" for x in Quantity)
    m = ", ".join(f"{x.value}" for x in Method)
    sf = ", ".join(f"{x.value}" for x in SampleForm)
    return (
        "Return JSON: {\"materials_studied\": [str], \"key_findings\": [str], \"parameters\": [\n"
        "  {\"material\": {\"formula\": str, \"stack\": str|null, \"doping\": str|null, "
        f"\"sample_form\": one of [{sf}], \"thickness_nm\": float|null, \"layers\": int|null, "
        "\"substrate\": str|null},\n"
        f"   \"quantity\": one of [{q}], \"quantity_label\": str, \"value\": float|null, "
        "\"value_min\": float|null, \"value_max\": float|null, \"uncertainty\": float|null, "
        "\"unit\": str, \"temperature_K\": float|null, \"field_T\": float|null,\n"
        f"   \"method\": one of [{m}], \"method_detail\": str|null, \"is_theoretical\": bool, "
        "\"evidence_quote\": str (verbatim from text), \"section\": str|null, \"confidence\": float 0-1}\n]}"
    )
