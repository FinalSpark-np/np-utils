"""
Neuroplatform Utils - Utility functions for the Neuroplatform project.

This is intended to be a plug-and-play utility module for V1, without any changes to the main Neuroplatform codebase.

StimParamLoader : Stimulation parameter loader and preview functions. Author : Cyril Achard, October 2024
"""

### IMPORTS ###
### StimParamLoader
from typing import List
import logging
from sys import stdout as STDOUT
from enum import Enum
from pathlib import Path
from PIL import Image
from collections import OrderedDict
from datetime import datetime, UTC
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass

try:
    from neuroplatform import StimParam, StimShape
except ModuleNotFoundError:
    raise ModuleNotFoundError("Neuroplatform access is required to use this utility.")

try:
    from neuroplatform import IntanSofware as IntanSoftware
except ImportError:
    try:
        from neuroplatform import IntanSoftware as IntanSoftware
    except ImportError:
        raise ImportError("No 'IntanSofware' or 'IntanSoftware'.")

### ENUMS ###


class MEA(Enum):
    """Mea Number"""

    One = 0
    Two = 1
    Three = 2
    Four = 3

    @staticmethod
    def get_from_electrode(electrode: int):
        return MEA(electrode // 32)


class Site(Enum):
    """Neurosphere ID, from 1 to 4"""

    One = 0
    Two = 1
    Three = 2
    Four = 3

    @staticmethod
    def get_from_electrode(electrode_id: int):
        site = (electrode_id % 32) // 8
        return Site(site)


### Setup and constants ###
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False

if not log.handlers:
    handler = logging.StreamHandler(STDOUT)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    log.addHandler(handler)


class LogHistory:
    """LogHistory - Utility class to manage log messages and display them in a DataFrame.

    Args:
        verbose (bool, optional): If True, display log messages of INFO level. Defaults to True.

    Methods:
        - info(message: str): Log an INFO message
        - warning(message: str): Log a WARNING message
        - error(message: str): Log an ERROR message
        - debug(message: str): Log a DEBUG message
        - get_history(): Return a DataFrame of all log messages
    """

    def __init__(self, verbose=True):
        self._history = OrderedDict()
        self.logger = log
        self.verbose = verbose

    def info(self, message):
        self._history[datetime.now(UTC)] = ("INFO", message)
        if self.verbose:
            self.logger.info(message)

    def warning(self, message):
        self._history[datetime.now(UTC)] = ("WARNING", message)
        self.logger.warning(message)

    def error(self, message):
        self._history[datetime.now(UTC)] = ("ERROR", message)
        self.logger.error(message)

    def debug(self, message):
        self._history[datetime.now(UTC)] = ("DEBUG", message)
        self.logger.debug(message)

    def get_history(self):
        return pd.DataFrame.from_dict(
            self._history, orient="index", columns=["Level", "Message"]
        )


class StimParamLoader:
    """StimParamLoader - Utility class to manage stimulation parameters and preview them in a plot.

    This utility can do the following for you :
    - Preview the stimulation parameters in a plot
    - Checks that all parameters are valid and enabled before sending them to the Intan
    - Send the parameters to the Intan software
    - Disable all parameters and send the update to the Intan, useful when you are done with the stimulation

    Args:
        stimparams (List[StimParam]): List of stimulation parameters. Cannot exceed 16 parameters.
        intan (IntanSoftware, optional): Intan software instance. Defaults to None.
        verbose (bool, optional): If True, display log messages. Defaults to True.
        must_connect (bool, optional): If True, raise an error if the Intan is not connected. Defaults to False.

    Methods:
        - add_stimparam(stimparam: StimParam): Append a new StimParam to the list of parameters.
        - preview_parameters(hide_empty: bool = True, colorblind: bool = False): Preview the parameters in a plot.
        - enable_all(): Enable all parameters.
        - disable_all(): Disable all parameters.
        - send_parameters(): Send the parameters to the Intan.
        - disable_all_and_send(): Disable all parameters and send them to the Intan.
        - show_all_stimparams(): Print out all the parameters.
        - all_parameters_enabled() -> bool: Check if all parameters are enabled.
        - reset(): Clear all parameters.
        - get_log(): Return a DataFrame of all log messages. Use to get a history of all messages even if verbose is False.

    Raises:
        ValueError: If the parameter is not a StimParam instance.
        ValueError: If the electrode is already in use within one of the supplied parameters.
        ValueError: If the trigger is already in use within one of the supplied parameters.
        RuntimeError: If the Intan is not connected and must_connect is True.
        FileNotFoundError: If the MEA schema image for plotting is not found.

    Example:
        # Initialize the loader with stimulation parameters
        stimparams = [
            StimParam(index=42, enable=True, ...), # add more parameters as needed
            ...
        ]
        loader = StimParamLoader(stimparams)

        # Preview the parameters
        loader.preview_parameters()

        # Send the parameters to the Intan software
        loader.send_parameters()

        # Disable all parameters and send the update
        loader.disable_all_and_send()
    """

    MEA_SCHEMA = str(Path(__file__).parent / "MEA_schema.png")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ModuleNotFoundError(
            "Plotly not found. Please install plotly and make sure to reload your kernel."
        )

    def __init__(
        self,
        stimparams: List[StimParam] | None,
        intan: IntanSoftware = None,
        verbose=True,
        must_connect=False,
    ):
        """Create a StimParamLoader instance.

        Args:
            - stimparams (List[StimParam]): List of stimulation parameters. Cannot exceed 16 parameters.
            - intan (IntanSoftware, optional): Intan software instance. Defaults to None.
            - verbose (bool, optional): If True, display log messages. Defaults to True.
            - must_connect (bool, optional): If True, raise an error if the Intan is not connected. Defaults to False.
        """
        self._stimparams = []
        self._used_electrodes = []
        self._used_triggers = []
        self._sites = {}
        self._meas = {}
        self._electrodes_location = {}
        self._mea_image = None
        self._site_coords_plot = {
            0: (9.14, 12.6),
            1: (7.5, 11),
            2: (9.14, 11),
            3: (9.14, 9.35),
            4: (10.72, 9.35),
            5: (10.72, 11),
            6: (12.31, 11),
            7: (10.72, 12.6),
        }
        ###
        self.log = LogHistory(verbose)
        self.log.info(
            "Please remember to book the system before connecting to the Intan."
        )
        self.intan = intan
        self.stimparams = stimparams

        if not Path(self.MEA_SCHEMA).exists():
            raise FileNotFoundError(
                f"MEA schema {self.MEA_SCHEMA} not found. Make sure the file is present on your machine, and that the path is correct."
            )
        else:
            self._mea_image = Image.open(self.MEA_SCHEMA)

        if self.intan is None:
            if must_connect:
                raise RuntimeError("Could not connect to Intan")
            else:
                self.log.warning(
                    "Could not connect to Intan. You may preview parameters but sending parameters to the Intan will not be possible."
                )

    @property
    def stimparams(self):
        return self._stimparams

    @stimparams.setter
    def stimparams(self, new_stimparams: List[StimParam]):
        # if len(stimparams) > 16:
        #     raise ValueError("Maximum number of parameters reached (16)")
        if new_stimparams is None:
            self._stimparams = []
            self.log.info("No parameters set.")
            return
        for param in new_stimparams:
            if not isinstance(param, StimParam):
                raise ValueError(f"{param} is not a StimParam instance")
        self._stimparams = new_stimparams
        self._update_parameters()

    def _update_parameters(self):
        if len(self._stimparams) == 0:
            return
        self._clear_records()
        for param in self.stimparams:
            if param.index in self._used_electrodes:
                raise ValueError(
                    f"Electrode {param.index} is already in use. Only one parameter per electrode is possible currently."
                )
            self._used_electrodes.append(param.index)
            # if param.trigger_key in self._used_triggers:
            # raise ValueError(
            # f"Trigger {param.trigger_key} is already in use. Using two parameters with the same trigger would overwrite the first one; please use a different trigger key."
            # )
            if param.trigger_key in self._used_triggers:
                self._used_triggers.append(param.trigger_key)
            if param.index < 0 or param.index > 127:
                raise ValueError(f"Invalid electrode number: {param.index}")
            if param.trigger_key < 0 or param.trigger_key > 15:
                raise ValueError(f"Invalid trigger key: {param.trigger_key}")
            if param.phase_duration1 < 0 or param.phase_duration2 < 0:
                raise ValueError(
                    f"Invalid phase duration: {param.phase_duration1}, {param.phase_duration2}"
                )
            if param.phase_amplitude1 < 0 or param.phase_amplitude2 < 0:
                raise ValueError(
                    f"Invalid phase amplitude: {param.phase_amplitude1}, {param.phase_amplitude2}"
                )
            if param.phase_duration1 > 500 or param.phase_duration2 > 500:
                self.log.warning(
                    f"Phase duration exceeds 500 us: {param.phase_duration1}, {param.phase_duration2}"
                )
            if (
                param.phase_duration1 * param.phase_amplitude1
                != param.phase_duration2 * param.phase_amplitude2
                and (
                    param.stim_shape == StimShape.Biphasic
                    or param.stim_shape == StimShape.BiphasicWithInterphaseDelay
                )
            ):
                self.log.warning(
                    f"Pulses are not charge balanced for electrode {param.index}. Please check that this is intentional; otherwise make sure that the product of the phase duration and amplitude are equal for both phases."
                )
            if (
                param.stim_shape == StimShape.Triphasic
                and param.phase_duration2 * param.phase_amplitude2 * 2
                != param.phase_duration1 * param.phase_amplitude1
            ):
                self.log.warning(
                    f"Pulses are not charge balanced for electrode {param.index}. Please check  that this is intentional; otherwise make sure that (phase2_duration * phase2_amplitude * 2) = (phase1_duration * phase1_amplitude)."
                )
            self._electrode_param_mapping[param.index] = param
            site = Site.get_from_electrode(param.index)
            mea = MEA.get_from_electrode(param.index)
            self._sites[param.index] = site
            self._meas[param.index] = mea
        if len(set(self._meas.values())) > 1:
            self.log.warning(
                "Parameters have been set across multiple MEAs. Please make sure this is intentional."
            )

    def _clear_records(self):
        self._used_electrodes = []
        self._used_triggers = []
        self._electrode_param_mapping = {}
        self._sites = {}
        self._meas = {}

    def get_log(self):
        """Return a DataFrame of all log messages.

        Use this to get a history of all messages even if verbose is False.
        """
        return self.log.get_history()

    def reset(self):
        """Clear all parameters."""
        self.stimparams = []

    def add_stimparam(self, stimparam: StimParam):
        """Append a new StimParam to the list of parameters.

        Args:
            stimparam (StimParam): The new parameter to add.
        """
        try:
            self.stimparams.append(stimparam)
            self._update_parameters()
            return True
        except Exception as e:
            self.log.error(f"Error: {e}")
            return False

    def show_all_stimparams(self):
        """Print out all the parameters."""
        if len(self.stimparams) == 0:
            self.log.info("No parameters to display.")
        for electrode, stimparam in self._electrode_param_mapping.items():
            self.log.info(f"Electrode {electrode}:\n{stimparam.display_attributes()}")
            self.log.info("*" * 50)

    def _plot_site(
        self,
        fig,
        row,
        col,
        stimparams_: List[stimparams],
        site: Site,
        mea: MEA,
        colorblind=False,
    ):
        fig.add_layout_image(
            source=self._mea_image,
            x=0,
            y=0,
            xref="x",
            yref="y",
            xanchor="left",
            yanchor="bottom",
            sizex=20,
            sizey=20,
            opacity=0.75,
            layer="below",
            sizing="stretch",
            row=row,
            col=col,
        )

        x_coords, y_coords, colors, tooltips, sizes = [], [], [], [], []
        used_sites = [
            param.index % 8 for param in stimparams_
        ]  # maps the electrode to one of the 8 electrodes of the site
        for electrode in range(8):
            if electrode in used_sites:
                param = stimparams_[used_sites.index(electrode)]
                x, y = self._site_coords_plot[electrode]
                x_coords.append(x)
                y_coords.append(y)
                if colorblind:
                    colors.append("blue" if param.enable else "yellow")
                else:
                    colors.append("#90EE90" if param.enable else "red")
                info = f"Electrode {param.index}: {'Enabled' if param.enable else 'Disabled'}"
                info += f"<br>Trigger : {param.trigger_key}"
                info += f"<br>Nb Pulse : {param.nb_pulse}"
                info += f"<br>Period : {param.pulse_train_period}"
                info += f"<br>Shape : {param.stim_shape}"
                info += f"<br>Polarity : {param.polarity}"
                info += f"<br>D1 : {param.phase_duration1} us"
                info += f"<br>A1 : {param.phase_amplitude1} uA"
                info += f"<br>D2 : {param.phase_duration2} us"
                info += f"<br>A2 : {param.phase_amplitude2} uA"
                tooltips.append(info)
                sizes.append(10)
            else:
                x, y = self._site_coords_plot[electrode]
                x_coords.append(x)
                y_coords.append(y)
                colors.append("black")
                sizes.append(5)
                # show electrode number based on all MEAs (from 0 to 127)
                tooltips.append(
                    f"'<span style=\"color:white\">Electrode {mea.value * 32 + site.value * 8 + electrode}</span>'"
                )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(size=sizes, color=colors),
                text=tooltips,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )

        fig.add_annotation(
            x=9.5,
            y=16,
            xref="x",
            yref="y",
            text=f"MEA {mea.value + 1}<br>Site {site.value + 1}",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center",
            yanchor="middle",
            row=row,
            col=col,
        )

        fig.update_xaxes(
            range=[0, 20],
            dtick=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            range=[0, 20],
            dtick=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            row=row,
            col=col,
        )

    def preview_parameters(self, hide_empty: bool = True, colorblind: bool = False):
        """Preview the parameters in a plot.

        Args:
            hide_empty (bool, optional): Hide empty sites. Defaults to True.
            colorblind (bool, optional): Use colorblind friendly colors. Defaults to False.
        """
        if len(self.stimparams) == 0:
            self.log.info("No parameters to display.")
            return

        mea_site_dict = {}
        for stimparam in self.stimparams:
            mea = MEA.get_from_electrode(stimparam.index)
            site = Site.get_from_electrode(stimparam.index)
            if mea not in mea_site_dict:
                mea_site_dict[mea] = {}
            if site not in mea_site_dict[mea]:
                mea_site_dict[mea][site] = []
            mea_site_dict[mea][site].append(stimparam)

        num_meas = len(mea_site_dict)
        num_sites_per_mea = 4

        fig = make_subplots(
            rows=num_meas,
            cols=num_sites_per_mea,
            horizontal_spacing=0.05,
            vertical_spacing=0.05,
        )
        self.log.info("Plotting...")
        self.log.info(
            "___ This preview does not reflect parameters currently set on the Intan ___"
        )
        for row, (mea, sites) in enumerate(mea_site_dict.items(), start=1):
            for col, site in enumerate(Site, start=1):
                stimparams_ = sites.get(site, [])
                if hide_empty and not stimparams_:
                    continue
                self._plot_site(fig, row, col, stimparams_, site, mea, colorblind)

        scaling = 0.5
        fig.update_layout(
            showlegend=False,
            height=350 * (num_meas + 1) * scaling,
            width=2000 * scaling,
            title={
                "text": "Parameter preview",
                "font": {"size": 20, "color": "white"},
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            paper_bgcolor="rgba(192,192,192,0.35)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        config = {"displayModeBar": False, "scrollZoom": False}
        fig.show(config=config)

    def all_parameters_enabled(self) -> bool:
        """Check if all parameters are enabled."""
        if len(self._stimparams) == 0:
            return False
        return all(param.enable for param in self.stimparams)

    def enable_all(self):
        """Enable all parameters."""
        if len(self._stimparams) == 0:
            self.log.warning("No parameters to enable.")
            return
        for param in self.stimparams:
            param.enable = True

    def disable_all(self):
        """Disable all parameters."""
        if len(self._stimparams) == 0:
            self.log.warning("No parameters to disable.")
            return
        for param in self.stimparams:
            param.enable = False

    def _send_parameters(self):
        if self.intan is None:
            raise ValueError("Intan not connected")
        if (
            len(self._stimparams) == 0
        ):  # not needed as we check this in the send_parameters method, but leaving it here for now
            self.log.warning("No parameters to send.")
            return
        self._update_parameters()
        self.intan.send_stimparam(self.stimparams)

    def send_parameters(self):
        """Send the parameters to the Intan.

        Returns:
            bool: True if the parameters were sent successfully, False otherwise.
        """
        try:
            if self.intan is None:
                raise ValueError("Intan not connected")
            if len(self._stimparams) == 0:
                self.log.warning("No parameters to send.")
                return
            if not self.all_parameters_enabled():
                self.log.warning(
                    "--- Some parameters are disabled. Please make sure to enable the parameters you want to use. ---"
                )

            self.log.info("Sending... Please wait 10 seconds")
            self.intan.send_stimparam(self.stimparams)
            self.log.info("Done.")
            return True
        except Exception as e:
            self.log.error(f"Error: {e}")
            return False

    def disable_all_and_send(self):
        """
        Disable all parameters and send them to the Intan.
        Use this when you are finished with the stimulation.

        Returns:
            bool: True if the parameters were disabled and sent successfully, False otherwise.
        """
        try:
            if len(self._stimparams) == 0:
                self.log.warning("No parameters to disable.")
                return False
            self.disable_all()
            self._send_parameters()
            self.log.info("All parameters disabled and sent to Intan.")
            return True
        except Exception as e:
            self.log.error(f"Error: {e}")
            return False
