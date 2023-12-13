"""
Run a cpymad MAD-X simulation for the LHC optics (2018),
assign measured errors from a WISE realization and correct
the nonlinear errors in the IR.
At the end output for SixTrack input is written.

The ``main()`` function set's up the beams and is responsible to assign the
IRNL corrections to the right optics. These are the things that make this
study specific.

The class ``LHCBeam`` is setting up and running cpymad.
This class can be useful for a lot of different studies, by extending
it with extra functionality.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Sequence, Union

import tfs
from cpymad.madx import Madx
from cpymad_lhc.coupling_correction import correct_coupling
from cpymad_lhc.general import (amplitude_detuning_ptc, deactivate_arc_sextupoles, get_k_strings,
                                get_lhc_sequence_filename_and_bv, get_tfs, match_tune,
                                power_landau_octupoles, switch_magnetic_errors)
from cpymad_lhc.ir_orbit import log_orbit, orbit_setup
from cpymad_lhc.logging import MADXCMD, MADXOUT, cpymad_logging_setup
from optics_functions.coupling import closest_tune_approach, coupling_via_cmatrix
from pandas import DataFrame
from tfs import TfsDataFrame

LOG = logging.getLogger(__name__)  # setup in main()
LOG_LEVEL = logging.DEBUG
ACC_MODELS = "acc-models-lhc"

PATHS = {
    # original afs paths ---
    "db5": Path("/afs/cern.ch/eng/lhc/optics/V6.503"),
    "optics2016": Path("/afs/cern.ch/eng/lhc/optics/runII/2016"),
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "optics_repo": Path("/afs/cern.ch/eng/acc-models/lhc"),
    "wise2011": Path("/afs/cern.ch/eng/lhc/optics/V6.503/WISE/After_sector_3-4_repair/"),
    "wise_jdilly": Path("/afs/cern.ch/work/j/jdilly/wise/"),
    "heads": Path("/afs/cern.ch/work/j/jdilly/study.b6_correction_with_heads"),
    # modified for this example ---
    # "db5": Path("./lhc_optics"),
    # "optics2016": Path("./lhc_optics"),
    # "optics2018": Path("./lhc_optics"),
    # "wise": Path("./lhc_optics/wise_2015"),
    ACC_MODELS: Path(ACC_MODELS),
}


def pathstr(key: str, *args: str) -> str:
    """ Wrapper to get the path (as string! Because MADX wants strings)
    with the base from the dict ``PATHS``.

    Args:
        key (str): Key for the base-path in ``PATHS``.
        args (str): Path parts to attach to the base.

    Returns:
        str: Full path with the base from  given ``key``.
    """
    return str(PATHS[key].joinpath(*args))


def get_optics_path(year: int, name: Union[str, Path]):
    """ Get optics by name, i.e. a collection of optics path-strings to the optics files.

     Args:
         year (int): Year of the optics
         name (str, Path): Name for the optics or a path to the optics file.

    Returns:
        str: Path to the optics file.
     """
    if isinstance(name, Path):
        return str(name)

    optics_map = {
        2018: {
            'inj': pathstr("optics2018", "PROTON", "opticsfile.1"),
            'flat6015': pathstr("optics2018", 'MDflatoptics2018', 'opticsfile_flattele60cm.21'),
            'round3030': pathstr("optics2018", "PROTON", "opticsfile.22_ctpps2"),
        },
        2022: {
            'round3030': pathstr(ACC_MODELS, "strengths", "ATS_Nominal", "2022", "squeeze", "ats_30cm.madx")
        }
    }
    return optics_map[year][name]


def get_wise_path(seed: int, year: int = 2015, optics: str = ""):
    """ Get the wise errordefinition file by seed-number.

    Args:
        seed (int): Seed for the error realization.
        yaer (int): Which wise year defined above? default 2015

    Returns:
        str: Path to the wise errortable file.
    """
    if year == 2011:
        wise_path = "wise2011"
        if optics == "inj":
            return pathstr(wise_path, "injection", f"injection_errors-emfqcs-{seed:d}.tfs")
        return pathstr(wise_path, "collision", f"collision_7000-emfqcs-{seed:d}.tfs")

    wise_path = "wise_jdilly"
    if year == 2015:
        if optics == "inj":
            raise NotImplementedError("Wise 2015 Injection optics not copied from DFS yet.")
        return pathstr(wise_path, "WISE-2015-LHCsqueeze-0.4_10.0_0.4_3.0-6.5TeV-emfqcs", f"WISE.errordef.{seed:04d}.tfs")

    if year == 2021:
        opticsfile_map = {
            "round3030": "opticsfile.32"
        }
        try:
            opticsfile = opticsfile_map[optics]
        except KeyError as e:
            raise NotImplementedError("Wise 2021 is only implemented for round3030 (i.e. ats_30cm).") from e
        
        return pathstr(wise_path, "WISE-2021-v6-body-only", opticsfile, "seeds", f"{opticsfile}-emfqcs-{seed:04d}.tfs")

    raise NotImplementedError("Only the wise years 2011, 2015 and 2021 are implemented.")


def drop_allzero_columns(df: TfsDataFrame) -> TfsDataFrame:
    """ Drop columns that contain only zeros, to save harddrive space.

    Args:
        df (TfsDataFrame): DataFrame with all data

    Returns:
        TfsDataFrame: DataFrame with only non-zero columns.
    """
    return df.loc[:, (df != 0).any(axis="index")]


def get_detuning_from_ptc_output(df: DataFrame, beam: int = None, log: bool = True) -> Dict[str, float]:
    """ Convert PTC amplitude detuning output to dict and log values.

    Args:
        df (DataFrame): DataFrame as given by PTC.
        beam (int): Beam used (for logging purposes only)
        log (bool): Print values to the logger

    Returns:
        dict[str, float]: Dictionary with entries 'X', 'Y', 'XY'
        with the values for the direct X, direct Y and cross Term respectively

    """
    map = {"X": "X10", "Y": "Y01", "XY": "X01"}
    results = {name: None for name in map.keys()}
    if log:
        LOG.info("Current Detuning Values" + ("" if not beam else f" in Beam {beam}"))
    for name, term in map.items():
        value = df.query(
            f'NAME == "ANH{term[0]}" and '
            f'ORDER1 == {term[1]} and ORDER2 == {term[2]} '
            f'and ORDER3 == 0 and ORDER4 == 0'
        )["VALUE"].to_numpy()[0]
        if log:
            LOG.info(f"  {name:<2s}: {value}")
        results[name] = value
    return results


@dataclass()
class Correction:
    """ DataClass to store correction data. """
    name: str
    df: TfsDataFrame
    cmd: str


#LHCBeam Dataclass -------------------------------------------------------------

@dataclass()
class LHCBeam:
    """ Object containing all the information about the machine setup and
    performing the MAD-X commands to run the simulation. """
    beam: int
    outputdir: Path
    xing: dict
    errors: dict
    optics: str
    correct_irnl: bool = True
    thin: bool = True
    seed: int = 1
    tune_x: float = 62.31
    tune_y: float = 60.32
    chroma: float = 3
    year: int = 2018
    wise: int = 2015
    emittance: float = 7.29767146889e-09
    n_particles: float = 1.0e10   # number of particles in beam
    on_arc_errors: bool = False  # apply field errors to arcs
    # Placeholders (set in functions)
    df_twiss_nominal: TfsDataFrame = field(init=False)
    df_twiss_nominal_ir: TfsDataFrame = field(init=False)
    df_ampdet_nominal: TfsDataFrame = field(init=False)
    df_errors: TfsDataFrame = field(init=False)
    df_errors_ir: TfsDataFrame = field(init=False)
    correction: Correction = field(init=False)
    df_twiss_corrected: TfsDataFrame = field(init=False)
    df_ampdet_corrected: TfsDataFrame = field(init=False)
    # Constants
    ACCEL: ClassVar[str] = 'lhc'
    TWISS_COLUMNS: ClassVar[Sequence[str]] = tuple(['NAME', 'KEYWORD', 'S', 'X', 'Y', 'L', 'LRAD',
                                                    'BETX', 'BETY', 'ALFX', 'ALFY',
                                                    'DX', 'DY', 'MUX', 'MUY',
                                                    'R11', 'R12', 'R21', 'R22'] + get_k_strings())
    ERROR_COLUMNS: ClassVar[Sequence[str]] = tuple(["NAME", "DX", "DY"] + get_k_strings())

    # Init ---

    def __post_init__(self):
        """ Setup the MADX, output dirs and logging as well as additional instance parameters. """
        self.outputdir.mkdir(exist_ok=True, parents=True)
        self.madx = Madx(**cpymad_logging_setup(level=LOG_LEVEL,  # sets also standard loggers
                                                command_log=self.outputdir/'madx_commands.log',
                                                full_log=self.outputdir/'full_output.log'))
        self.logger = {key: logging.getLogger(key).handlers for key in ("", MADXOUT, MADXCMD)}  # save logger to reinstate later
        self.madx.globals.mylhcbeam = self.beam                # used in macros
        self.madx.globals.bv_aux = -1 if self.beam > 2 else 1  # can be used in macros

        # Define Sequence to use
        # lhc after 2020 behaves like hl-lhc with regard to file-names
        self.seq_name, self.seq_file, self.bv_flag = get_lhc_sequence_filename_and_bv(self.beam, accel="lhc" if self.year < 2020 else "hllhc")
        if self.correct_irnl and not self.thin:
            raise NotImplementedError("To correct IRNL errors a thin lattice is required.")

    # Output Helper ---

    def output_path(self, type_: str, output_id: str, dir_: Path = None, suffix: str = ".tfs") -> Path:
        """ Returns the output path for standardized tfs names in the default output directory.

        Args:
            type_ (str): Type of the output file (e.g. 'twiss', 'errors', 'ampdet')
            output_id (str): Name of the output (e.g. 'nominal')
            dir_ (Path): Override default directory.
            suffix (str): suffix of the output file.

        Returns:
            Path: Path to the output file
         """
        if dir_ is None:
            dir_ = self.outputdir
        return dir_ / f'{type_}.lhc.b{self.beam:d}.{output_id}{suffix}'

    def get_twiss(self, output_id=None, index_regex=r"BPM|M|IP", **kwargs) -> TfsDataFrame:
        """ Uses the ``twiss`` command to get the current optics in the machine
        as TfsDataFrame.

        Args:
            output_id (str): ID to use in the output (see ``output_path``).
                             If not given, no output is written.
            index_regex (str): Filter DataFrame index (NAME) by this pattern.

        Returns:
            TfsDataFrame: DataFrame containing the optics.
        """
        kwargs['chrom'] = kwargs.get('chrom', True)
        self.madx.twiss(sequence=self.seq_name, **kwargs)
        df_twiss = self.get_last_twiss(index_regex=index_regex)
        if output_id is not None:
            self.write_tfs(df_twiss, 'twiss', output_id)
        return df_twiss

    def get_last_twiss(self, index_regex=r"BPM|M|IP") -> TfsDataFrame:
        """ Returns the twiss table of the last calculated twiss.

        Args:
            index_regex (str): Filter DataFrame index (NAME) by this pattern.

        Returns:
            TfsDataFrame: DataFrame containing the optics.
        """
        return get_tfs(self.madx.table.twiss, columns=self.TWISS_COLUMNS, index_regex=index_regex)

    def get_errors(self, output_id: str = None, index_regex: str = "M") -> TfsDataFrame:
        """ Uses the ``etable`` command to get the currently assigned errors in the machine
        as TfsDataFrame.

        Args:
            output_id (str): ID to use in the output (see ``output_path``).
                             If not given, no output is written.
            index_regex (str): Filter DataFrame index (NAME) by this pattern.

        Returns:
            TfsDataFrame: DataFrame containing errors.
        """
        # As far as I can tell `only_selected` does not work with
        # etable and there is always only the selected items in the table
        # (jdilly, cpymad 1.4.1)
        self.madx.select(flag='error', clear=True)
        self.madx.select(flag='error', column=self.ERROR_COLUMNS)
        self.madx.etable(table='error')
        df_errors = get_tfs(self.madx.table.error, index_regex=index_regex, columns=self.ERROR_COLUMNS)
        if output_id is not None:
            self.write_tfs(df_errors, 'errors', output_id)
        return df_errors

    def get_ampdet(self, output_id: str) -> TfsDataFrame:
        """ Write out current amplitude detuning via PTC.

        Args:
            output_id (str): ID to use in the output (see ``output_path``).
                             If not given, no output is written.

        Returns:
            TfsDataFrame: Containing the PTC output data.
        """
        file = None
        if output_id is not None:
            file = self.output_path('ampdet', output_id)
            LOG.info(f"Calculating amplitude detuning for {output_id}.")
        df_ampdet = amplitude_detuning_ptc(self.madx, ampdet=2, chroma=4, file=file)
        get_detuning_from_ptc_output(df_ampdet, beam=self.beam)
        return df_ampdet

    def write_tfs(self, df: TfsDataFrame, type_: str, output_id: str):
        """ Write the given TfsDataFrame with the standardized name (see ``output_path``)
        and the index ``NAME``.

        Args:
            df (TfsDataFrame): DataFrame to write.
            type_ (str): Type of the output file (see ``output_path``)
            output_id (str): Name of the output (see ``output_path``)
        """
        tfs.write(self.output_path(type_, output_id), drop_allzero_columns(df), save_index="NAME")

    # Wrapper ---

    def log_orbit(self):
        """ Log the current orbit. """
        log_orbit(self.madx, accel=self.ACCEL)

    def closest_tune_approach(self, df: TfsDataFrame = None):
        """ Calculate and print out the closest tune approach from the twiss
        DataFrame given. If no frame is given, it gets the current twiss.

        Args:
            df (TfsDataFrame): Twiss DataFrame.
        """
        if df is None:
           df = self.get_twiss()
        df_coupling = coupling_via_cmatrix(df)
        closest_tune_approach(df_coupling, qx=self.tune_x, qy=self.tune_y)

    def correct_coupling(self):
        """ Correct the current coupling in the machine. """
        correct_coupling(self.madx,
                         accel=self.ACCEL, sequence=self.seq_name,
                         qx=self.tune_x, qy=self.tune_y,
                         dqx=self.chroma, dqy=self.chroma)

    def match_tune(self):
        """ Match the machine to the preconfigured tunes. """
        match_tune(self.madx,
                   accel=self.ACCEL, sequence=self.seq_name,
                   qx=self.tune_x, qy=self.tune_y,
                   dqx=self.chroma, dqy=self.chroma)

    def apply_measured_errors(self, *magnets: str):
        """ Apply the measured errors to the given magnets via Efcomp-files.

        Args:
            magnets (Sequence[str]): Names of the magnets to apply the errors to.
                                     (As in the Efcomp filenames).
        """
        for magnet in magnets:
            self.madx.call(pathstr('db5', 'measured_errors', f'Efcomp_{magnet}.madx'))

    def reinstate_loggers(self):
        """ Set the saved logger handlers to the current logger. """
        for name, handlers in self.logger.items():
            logging.getLogger(name).handlers = handlers

    def get_other_beam(self):
        """ Return the respective other beam number. """
        return 1 if self.beam == 4 else 4

    # Main ---

    def setup_machine(self):
        """ Nominal machine setup function.
        Initialized the beam and applies optics, crossing. """
        self.reinstate_loggers()
        madx = self.madx  # shorthand
        mvars = madx.globals  # shorthand

        # Load Macros
        madx.call(pathstr("optics2018", "toolkit", "macro.madx"))

        # Lattice Setup ---------------------------------------
        # Load Sequence
        if self.year > 2019:
            acc_models_path = PATHS[ACC_MODELS]
            acc_models_path.unlink(missing_ok=True)
            acc_models_path.symlink_to(pathstr("optics_repo", str(self.year)))
            madx.call(pathstr(ACC_MODELS, self.seq_file))
        else:
            madx.call(pathstr("optics2018", self.seq_file))

        # Slice Sequence
        if self.thin:
            mvars.slicefactor = 4
            madx.beam()
            madx.call(pathstr("optics2018", "toolkit", "myslice.madx"))
            madx.beam()
            madx.use(sequence=self.seq_name)
            madx.makethin(sequence=self.seq_name, style="teapot", makedipedge=True)

        # Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
        madx.seqedit(sequence=self.seq_name)
        madx.flatten()
        madx.cycle(start="IP3")
        madx.endedit()

        # Define Optics
        madx.call(get_optics_path(year=self.year, name=self.optics))
        if self.optics == 'inj':
            mvars.NRJ = 450.000  # not defined in injection optics.1 but in the others
        
        # Install Fringe Placeholder (for HEADS)
        madx.call(file=pathstr('heads', 'install_mqxa_fringenl.madx'))

        # Make Beam 
        madx.beam(sequence=self.seq_name, bv=self.bv_flag,
                  energy="NRJ", particle="proton", npart=self.n_particles,
                  kbunch=1, ex=self.emittance, ey=self.emittance)

        # Setup Orbit
        orbit_vars = orbit_setup(madx, accel='lhc', **self.xing)

        madx.use(sequence=self.seq_name)

        # Save Nominal
        self.match_tune()
        self.df_twiss_nominal = self.get_twiss('nominal')
        self.df_ampdet_nominal = self.get_ampdet('nominal')
        self.log_orbit()

        # Save nominal optics in IR+Correctors for ir nl correction
        self.df_twiss_nominal_ir = self.get_last_twiss(index_regex="M(QS?X|BX|BRC|C[SOT]S?X)")
        self.write_tfs(self.df_twiss_nominal_ir, 'twiss', 'optics_ir')

    def apply_errors(self):
        """ Apply the errors onto the machine. The state is uncorrected afterwards. """
        self.reinstate_loggers()
        madx = self.madx  # shorthand

        if self.df_twiss_nominal is None:
            raise EnvironmentError("The machine needs to be setup first, before applying errors.")

        # Call error subroutines and measured error table for nominal LHC
        #
        # 'rotations_Q2_integral.tab', 'macro_error.madx' and 'Orbit_Routines.madx' are
        # called twice. It was like that in the mask I got from Ewen.
        # Possibly not neccessary. (jdilly, 2022-03-22)
        madx.call(file=pathstr('optics2016', 'measured_errors', 'Msubroutines.madx'))
        madx.readtable(file=pathstr('optics2016', 'measured_errors', 'rotations_Q2_integral.tab'))
        madx.call(file=pathstr('optics2016', 'errors', 'macro_error.madx'))  # some macros for error generation
        madx.call(file=pathstr('optics2016', 'toolkit', 'Orbit_Routines.madx'))
        madx.call(file=pathstr('optics2016', 'measured_errors', 'Msubroutines_new.madx'))  # think the new subroutines are only relevant for MSS - not used pre-2017 so shouldn't make a difference compared to old Msubroutines...
        madx.call(file=pathstr('optics2016', 'measured_errors', 'Msubroutines_MS_MSS_MO_new.madx'))
        madx.call(file=pathstr('optics2016', 'toolkit', 'Orbit_Routines.madx'))  # 2nd time
        madx.call(file=pathstr('optics2016', 'toolkit', 'SelectLHCMonCor.madx'))
        madx.readtable(file=pathstr('optics2016', 'measured_errors', 'rotations_Q2_integral.tab'))  # 2nd time
        madx.call(file=pathstr('optics2016', 'errors', 'macro_error.madx'))  # 2nd time

        # Apply magnetic errors -------------------------------
        switch_magnetic_errors(madx, **self.errors)

        # Read WISE ---
        madx.readtable(file=get_wise_path(self.seed, year=self.wise, optics=self.optics))

        # Read errors for HEADS ---
        madx.call(file=pathstr('heads', 'ITMQXAnc_errortable_all'))
        madx.call(file=pathstr('heads', 'ITMQXAcs_errortable_all'))

        # Apply errors to elements ---
        if self.on_arc_errors:
            self.apply_measured_errors('MB', 'MQ')

        self.apply_measured_errors(
            # IR Dipoles
            'MBXW',  # D1 in IP1 and IP5
            'MBRC',  # D2
            'MBX',  # D in IP2 and 8
            'MBRB',  # IP4
            'MBRS',  # IP4
            'MBW',  # IP7 and IP3
            # IR Quads
            'MQX',
            'MQY',
            'MQM',
            'MQMC',
            'MQML',
            'MQTL',
            'MQW',
        )

        # Apply errors for HEADS ---
        madx.call(file=pathstr('heads', 'Efcomp_MQXends.madx'))

        # Save uncorrected
        if not self.correct_irnl:
            self.closest_tune_approach()
            self.correct_coupling()

        self.match_tune()
        df_twiss_uncorrected = self.get_twiss('uncorrected')
        df_ampdet_uncorrected = self.get_ampdet('uncorrected')
        self.df_errors = self.get_errors('all')
        self.closest_tune_approach(df_twiss_uncorrected)

        # Save errors to table to be used for correction ---------------------------
        self.df_errors_ir = self.get_errors('ir', index_regex=r"M([QB]X|BRC)")

    def exit(self):
        """ End attached cpymad MADX instance. """
        PATHS[ACC_MODELS].unlink(missing_ok=True)
        self.reinstate_loggers()
        self.madx.exit()



# Main function ----------------------------------------------------------------

def main(beam: int,
         outputdir: Path,
         xing: dict = None,  # set to {'scheme': 'top'} below
         errors: dict = None,  # set to {'default': True} below
         optics: str = 'round3030',  # 30cm round optics
         seed: int = 1,
         year: int = 2018,
         wise: int = 2015,
         mo_current: int = 433,  # Powering of the landau octupoles in A
         ):
    """ Main function to run this script.
    First sets up the LHC machine for the beams defined by the `outputdirs` dict.
    Then runs the IRNL correction and assig

    Args:
        beam (int): Beam number
        outputdir (Path): Output directory
        xing (dict): Crossing scheme definition. See ``cpymad_lhc.ir_orbit.orbit_setup``
        errors (dict): Error definitions. See ``cpymad_lhc.general.switch_magnetic_errors``
        optics (str): Optics to use. See ``get_optics_path``.
        seed (int): Error realization seed for WISE tables. Between 1-60.
        year (int): Year of the model to use. If < 2020, 2018 optics are used.

    """
    # set mutable defaults ----
    if xing is None:
        xing = {'scheme': 'flat'}  # use top-energy crossing scheme

    if errors is None:
        errors = {f"AB{i}": True for i in range(3, 16)}  # activates default errors

    # Setup LHC for both beams -------------------------------------------------
    lhc_beam = LHCBeam(
        beam=beam, outputdir=outputdir,
        xing=xing, errors=errors, optics=optics,
        seed=seed,
        year=year,
        wise=wise,
        thin=False,
        correct_irnl=False,
    )
    lhc_beam.setup_machine()
    if errors:
        lhc_beam.apply_errors()

    # MO Powering --------------------------------------------------------------

    # deactivate sextupoles to  only little detuning a 0 power
    deactivate_arc_sextupoles(
        madx=lhc_beam.madx, 
        beam=lhc_beam.beam
    )

    # make sure they are off
    power_landau_octupoles(
        madx=lhc_beam.madx, 
        beam=lhc_beam.beam, 
        mo_current=0,
        defective_arc=False,
    )

    lhc_beam.match_tune()
    df_twiss_mo_none = lhc_beam.get_twiss(f'mo{0}')
    df_ampdet_mo_none = lhc_beam.get_ampdet(f'mo{0}')

    power_landau_octupoles(
        madx=lhc_beam.madx, 
        beam=lhc_beam.beam, 
        mo_current=mo_current,
        defective_arc=False,
    )

    lhc_beam.match_tune()
    df_twiss_mo_powered = lhc_beam.get_twiss(f'mo{mo_current}')
    df_ampdet_mo_powered = lhc_beam.get_ampdet(f'mo{mo_current}')

    unpowered = get_detuning_from_ptc_output(df_ampdet_mo_none)
    powered = get_detuning_from_ptc_output(df_ampdet_mo_powered)
    diff = {k: pow - unpow for k, unpow, pow in zip(unpowered.keys(), unpowered.values(), powered.values())}
    LOG.info(f"Detuning change with MO = {mo_current}A:")
    for name, value in diff.items():
            LOG.info(f"  {name:<2s}: {value}")


    # End MAD-X instances ------------------------------------------------------
    lhc_beam.exit()


if __name__ == '__main__':
    for beam in (1, 2):
        main(beam=beam, outputdir=Path(f"results/b{beam}"), year=2022, errors={}, mo_current=433)
