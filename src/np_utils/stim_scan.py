from time import sleep
from datetime import datetime, timedelta, UTC
import numpy as np
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from enum import Enum
from itertools import product

from neuroplatform import (
    Database,
    Trigger,
    StimParam,
    Experiment,
    StimPolarity,
    StimShape,
)

try:
    from neuroplatform import IntanSofware as IntanSoftware
except ImportError:
    try:
        from neuroplatform import IntanSoftware as IntanSoftware
    except ImportError:
        raise ImportError("No IntanSoftware or IntanSofware")

from .parameters_loader import StimParamLoader


class ExtendedTimedelta(timedelta):
    """Minimal extension of the timedelta class to add simple time unit conversion methods."""

    def as_minutes(self) -> float:
        """Returns the total number of minutes."""
        return self.total_seconds() / 60

    def as_seconds(self) -> float:
        """Returns the total number of seconds."""
        return self.total_seconds()

    def as_milliseconds(self) -> float:
        """Returns the total number of milliseconds."""
        return self.total_seconds() * 1e3

    def as_microseconds(self) -> float:
        """Returns the total number of microseconds."""
        return self.total_seconds() * 1e6

    def to_hertz(self) -> float:
        """Returns the frequency in hertz."""
        if self.total_seconds() == 0:
            print("Period is zero. Returning np.inf Hz.")
            return np.inf
        return 1 / self.total_seconds()


class MEAType(Enum):
    """Layout of the MEA."""

    MEA4x8 = 1
    MEA32 = 2

    def get_sites(self) -> int:
        if self == MEAType.MEA4x8:
            return 4
        elif self == MEAType.MEA32:
            return 1
        else:
            raise ValueError("MEA type not recognized.")

    def get_electrodes_per_site(self) -> int:
        if self == MEAType.MEA4x8:
            return 8
        elif self == MEAType.MEA32:
            return 32
        else:
            raise ValueError("MEA type not recognized.")


class MEA(
    Enum
):  # NOTE : these should likely be centralized and standardized as they are used in multiple places
    """MEA Number"""

    One = 0
    Two = 1
    Three = 2
    Four = 3

    @staticmethod
    def get_from_electrode(electrode: int) -> "MEA":
        return MEA(electrode // 32)

    @staticmethod
    def get_electrode_range(mea_number: "MEA") -> list[int]:
        return list(range(mea_number * 32, (mea_number + 1) * 32))


class Site(
    Enum
):  # NOTE : these should likely be centralized and standardized as they are used in multiple places
    """Neurosphere site ID, from 1 to 4"""

    One = 0
    Two = 1
    Three = 2
    Four = 3

    def get_from_electrode(electrode_id: int):
        site = (electrode_id % 32) // 8
        return Site(site)


@dataclass
class StimParamGrid:
    """Contains lists of all parameters to scan.

    Attributes:
        amplitudes: list[float]
            List of amplitudes to scan. Recommended to stay between 0.1 and 5.
        durations: list[float]
            List of durations to scan. Recommended to stay between 10 and 400.
        polarities: list[StimPolarity]
            List of polarities to scan. Accepted values are StimPolarity.NegativeFirst and StimPolarity.PositiveFirst.
        interphase_delays: list[float]
            List of interphase delays to scan. How long to wait between the end of the first phase and the start of the second phase.
        nb_pulses: list[int]
            List of number of pulses to scan. Will creat a spike train with the period specified in pulse_train_periods.
        pulse_train_periods: list[float]
            List of pulse train periods to scan. No effect if nb_pulses is 1.
        post_stim_ref_periods: list[float]
            List of post stimulation refractory periods to scan. Affects the time after a stimulation where no other stimulation can be sent.
        stim_shapes: list[StimShape]
            List of stimulation shapes to scan. Accepted values are StimShape.Biphasic and StimShape.BiphasicWithInterphaseDelay.
    """

    amplitudes: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    polarities: list[StimPolarity] = field(default_factory=list)
    interphase_delays: list[float] = field(default_factory=list)
    nb_pulses: list[int] = field(default_factory=list)
    pulse_train_periods: list[float] = field(default_factory=list)
    post_stim_ref_periods: list[float] = field(default_factory=list)
    stim_shapes: list[StimShape] = field(default_factory=list)
    mea_type: MEAType = MEAType.MEA4x8

    def __post_init__(self):
        attributes = {
            "amplitudes": (int, float),
            "durations": (int, float),
            "interphase_delays": (int, float),
            "nb_pulses": int,
            "pulse_train_periods": (int, float),
            "post_stim_ref_periods": (int, float),
            "stim_shapes": StimShape,
        }

        for attr, types in attributes.items():
            if not all(isinstance(item, types) for item in getattr(self, attr)):
                raise ValueError(f"All items in {attr} must be of type {types}.")

        if not isinstance(self.mea_type, MEAType):
            raise ValueError("MEA type must be a MEAType object.")

        if any(shape == StimShape.Triphasic for shape in self.stim_shapes):
            raise NotImplementedError(
                "Triphasic stimulation is not supported by this utility currently."
            )
            # this is because charge balancing works a bit differently for triphasic,
            # in order to release this I will leave it out for now

        default_param = StimParam()
        defaults = {
            "amplitudes": default_param.phase_amplitude1,
            "durations": default_param.phase_duration1,
            "polarities": default_param.polarity,
            "interphase_delays": default_param.interphase_delay,
            "nb_pulses": default_param.nb_pulse,
            "pulse_train_periods": default_param.pulse_train_period,
            "post_stim_ref_periods": default_param.post_stim_ref_period,
            "stim_shapes": default_param.stim_shape,
        }

        for attr, default in defaults.items():
            if not getattr(self, attr):
                setattr(self, attr, [default])

    def total_combinations(self) -> int:
        """Returns the total number of combinations."""
        total_combinations = 1
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, list) and not attr_name.startswith("_"):
                total_combinations *= len(attr)
        return total_combinations

    def display_grid(self):
        """Prints all the parameters in the grid."""
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                print(f"{k}: {v}")


@dataclass
class StimParamFactory:
    """Factory class to create StimParam objects from the grid."""

    amplitude1: float
    amplitude2: float
    duration1: float
    duration2: float
    polarity: StimPolarity
    interphase_delay: float
    nb_pulse: int
    pulse_train_period: float
    post_stim_ref_period: float
    stim_shape: StimShape

    def create_from(self):
        p = StimParam()
        p.phase_amplitude1 = self.amplitude1
        p.phase_amplitude2 = self.amplitude2
        p.phase_duration1 = self.duration1
        p.phase_duration2 = self.duration2
        p.polarity = self.polarity
        p.interphase_delay = self.interphase_delay
        p.nb_pulse = self.nb_pulse
        p.pulse_train_period = self.pulse_train_period
        p.post_stim_ref_period = self.post_stim_ref_period
        p.stim_shape = self.stim_shape
        return p

    def get_names(self):
        return {
            "Amplitude": self.amplitude1,
            "Duration": self.duration1,
            "Polarity": self.polarity,
            "Interphase delay": self.interphase_delay,
            "Number of pulses": self.nb_pulse,
            "Pulse train period": self.pulse_train_period,
            "Post stim ref period": self.post_stim_ref_period,
            "Stim shape": self.stim_shape,
        }

    def display_params(self):
        for k, v in self.get_names().items():
            print(f"- {k}: {v}")


class StimScan:
    def __init__(
        self,
        fs_experiment: Experiment,
        parameter_grid: StimParamGrid,
        scan_channels,
        delay_btw_stim,  # in seconds
        delay_btw_channels,  # in seconds
        repeats_per_channel,
    ):
        """Creates a stimulation parameter scan utility.

        Args:
        fs_experiment: Experiment
            The experiment object to use for the stimulation. Provided by neuroplatform API, requires a token.
        parameter_grid: StimParamGrid
            The grid of parameters to scan with. See StimParamGrid for more information.
        scan_channels: list[int]
            The list of channels to scan. Must be part of electrodes listed in 'fs_experiment.electrodes'.
        delay_btw_stim: float
            The delay between each stimulation in seconds. Note that if you are using pulse trains, this should be larger than the duration of the full pulse train.
        delay_btw_channels: float
            The delay between each channel stimulation in seconds. Use a higher value if you are concerned about fatigue or cross-talk.
        """
        self.fs_experiment = fs_experiment
        self.scan_channels = scan_channels
        self.delay_btw_stim = ExtendedTimedelta(seconds=delay_btw_stim)
        self.delay_btw_channels = ExtendedTimedelta(seconds=delay_btw_channels)
        self.repeats_per_channel = repeats_per_channel
        self.parameter_grid = parameter_grid

        self.params_per_electrode = self.parameter_grid.total_combinations()
        self.parameters = self._create_parameters_factory()
        self.mea_type = self.parameter_grid.mea_type
        self.mea = None
        self.loaders = None

        self.start_time = None
        self.stop_time = None

        self._trigger_gen = Trigger()
        self._intan = IntanSoftware()
        self._db = Database()

        self._channels_per_trigger = {}
        self._current_factory_id = None
        self._stim_history = OrderedDict()
        self._params_per_site = {}

        if not np.all(np.isin(scan_channels, fs_experiment.electrodes)):
            raise ValueError(
                "Some channels are not in the allowed electrodes list for your experiment token."
            )

        for channel in scan_channels:
            site = Site.get_from_electrode(channel)
            if site not in self._params_per_site:
                self._params_per_site[site] = 0
            self._params_per_site[site] += 1
            if self._params_per_site[site] > self.mea_type.get_electrodes_per_site():
                raise ValueError(
                    f"Too many provided channels for site {site}. Are all channels on the same MEA?"
                )

        mea = MEA.get_from_electrode(scan_channels[0]).value
        if not all(
            MEA.get_from_electrode(channel).value == mea for channel in scan_channels
        ):
            raise ValueError("All channels must be on the same MEA.")
        self.mea = mea

    def get_stimulation_parameter_history(self):
        """Returns a DataFrame of all the stimulation parameters sent."""
        return pd.DataFrame.from_dict(self._stim_history, orient="index")

    def _get_param_indices_by_trigger(self, trigger_key, loader):
        channels = []
        for param in loader.stimparams:
            if param.trigger_key == trigger_key:
                channels.append(param.index)
        return channels

    def _create_parameters_factory(self):
        parameters_factories = {}
        for i, combination in enumerate(
            product(
                self.parameter_grid.amplitudes,
                self.parameter_grid.durations,
                self.parameter_grid.polarities,
                self.parameter_grid.interphase_delays,
                self.parameter_grid.nb_pulses,
                self.parameter_grid.pulse_train_periods,
                self.parameter_grid.post_stim_ref_periods,
                self.parameter_grid.stim_shapes,
            )
        ):
            amp, dur, pol, ipd, nbp, ptp, psrp, ss = combination
            parameters_factories[i] = StimParamFactory(
                amplitude1=amp,
                amplitude2=amp,
                duration1=dur,
                duration2=dur,
                polarity=pol,
                interphase_delay=ipd,
                nb_pulse=nbp,
                pulse_train_period=ptp,
                post_stim_ref_period=psrp,
                stim_shape=ss,
            )
        return parameters_factories

    def _bind_parameter(self, params, factory, trigger_key, trigger_counter, index):
        p = factory.create_from()
        p.trigger_key = trigger_key
        p.index = index
        p.enable = True

        if p.index in self.scan_channels:
            trigger_counter[p.trigger_key] = 1
            params.append(p)
            if p.trigger_key not in self._channels_per_trigger:
                self._channels_per_trigger[p.trigger_key] = [p.index]
            else:
                self._channels_per_trigger[p.trigger_key].append(p.index)

    def _make_parameters(self):
        factory = self.parameters[self._current_factory_id]
        triggers_counter = np.zeros(16)

        if self.mea_type == MEAType.MEA4x8:
            params = []
            for site in range(self.mea_type.get_sites()):
                for trigger_key in range(self.mea_type.get_electrodes_per_site()):
                    self._bind_parameter(
                        params,
                        factory,
                        trigger_key,
                        triggers_counter,
                        self.mea * 32 + site * 8 + trigger_key,
                    )
            needed_triggers = np.where(triggers_counter > 0)[0]
            self.loaders = [
                (
                    needed_triggers,
                    StimParamLoader(params, self._intan, verbose=False),
                )
            ]
        elif self.mea_type == MEAType.MEA32:
            params_16 = []
            params_32 = []
            for trigger_key in range(
                16
            ):  # trigger_key cannot exceed 15, so we have to split this set in two loaders
                self._bind_parameter(
                    params_16,
                    factory,
                    trigger_key,
                    triggers_counter,
                    self.mea * 32 + trigger_key,
                )
                self._bind_parameter(
                    params_32,
                    factory,
                    trigger_key,
                    triggers_counter,
                    self.mea * 32 + 16 + trigger_key,
                )
            needed_triggers = np.where(triggers_counter > 0)[0]
            self.loaders = [
                (
                    needed_triggers,
                    StimParamLoader(params_16, self._intan, verbose=False),
                ),
                (
                    needed_triggers,
                    StimParamLoader(params_32, self._intan, verbose=False),
                ),
            ]

    def _send_stim(self):
        for needed_triggers, loader in self.loaders:
            loader.send_parameters()
            for trigger in needed_triggers:
                triggers = np.zeros(16, dtype=np.uint8)
                triggers[trigger] = 1
                for _ in range(self.repeats_per_channel):
                    params_data_dict = asdict(self.parameters[self._current_factory_id])
                    params_data_dict["trigger_key"] = trigger
                    params_data_dict["param_id"] = self._current_factory_id
                    params_data_dict["channel"] = self._get_param_indices_by_trigger(
                        trigger, loader
                    )
                    self._stim_history[datetime.now(UTC)] = params_data_dict
                    self._trigger_gen.send(triggers)
                    sleep(self.delay_btw_stim.as_seconds())
                sleep(self.delay_btw_channels.as_seconds())
            loader.disable_all_and_send()

    def run(self):
        """Runs the stimulation scan."""
        try:
            if self.fs_experiment.start():
                self.start_time = datetime.now(UTC)
                for factory_id in tqdm(self.parameters):
                    self._current_factory_id = factory_id
                    self._make_parameters()
                    self._send_stim()
        finally:
            if self.loaders is not None:
                for _, loader in self.loaders:
                    loader.disable_all_and_send()
            self._trigger_gen.close()
            self._intan.close()
            self.fs_experiment.stop()
            self.stop_time = datetime.now(UTC)

    def _plot_raster_for_channels(
        self,
        channel_df,
        show_electrodes=None,
        s_before=60,
        s_after=5,
        param_dict=None,
        guideline_freq=None,
        exp_name=None,
    ):
        channel_df = channel_df.copy()
        if show_electrodes is None:
            show_electrodes = self.scan_channels
        # prepare the dataframe for plotting
        channel_df.index.name = "Time"
        channel_df = channel_df.reset_index(drop=False)
        channel_df["Time"] = pd.to_datetime(channel_df["Time"])
        channel_df = channel_df.sort_values(by="Time")
        _, ax = plt.subplots(figsize=(20, 10))
        first_stim = channel_df["Time"].iloc[0]
        last_stim = channel_df["Time"].iloc[-1]
        y_axis_labs = MEA.get_electrode_range(self.mea)
        ax.set_yticks(ticks=range(len(y_axis_labs)), labels=y_axis_labs)
        offset = 0.5
        ax.set_ylim(-offset, len(y_axis_labs) - offset)
        ax.set_xlim(
            first_stim - timedelta(seconds=s_before),
            last_stim + timedelta(seconds=s_after),
        )
        # plot eventplot of all spike events for all channels on the y axis, and show vertical lines for each stim
        events = self._db.get_spike_event(
            first_stim - timedelta(seconds=s_before),
            last_stim + timedelta(seconds=s_after),
            self.fs_experiment.exp_name if exp_name is None else exp_name,
        )
        if events.empty:
            raise ValueError("No events for the selected time in database")
        # events = events[events["channel"].isin(show_electrodes)]
        # add a vertical line every guideline_freq seconds aligned on the first stim to visualize previous activity
        if guideline_freq is not None:
            try:
                for t, time_ in enumerate(
                    pd.date_range(
                        first_stim - timedelta(seconds=s_before),
                        last_stim + timedelta(seconds=s_after),
                        freq=guideline_freq,
                    )
                ):
                    ax.axvline(
                        time_,
                        color="yellow",
                        linestyle="--",
                        label=f"{guideline_freq} reference grid" if t == 0 else None,
                    )
            except ValueError:
                print(
                    "Warning : invalid guideline_freq. Format must be a string like '1s'. See pandas.date_range documentation for 'freq' argument."
                )
        # show stim times
        for t, time_ in enumerate(channel_df["Time"]):
            ax.axvline(
                time_,
                color="r",
                linestyle="--",
                label="Stimulation" if t == 0 else None,
            )

        # highlight the stim channel
        for channel in channel_df["channel"].values[0]:
            ax.fill_betweenx(
                [
                    channel - 0.25 - min(y_axis_labs),
                    channel + 0.25 - min(y_axis_labs),
                ],
                first_stim,
                last_stim,
                color="blue",
                alpha=0.3,
            )
        # raster plot
        events_channels = events["channel"].unique()
        for i, electrode in enumerate(y_axis_labs): 
            if electrode in events_channels:
                spikes = events[events["channel"] == electrode]["Time"]
                ax.eventplot(
                    spikes,
                    lineoffsets=i, 
                    linelengths=0.5,
                    color="blue",
                )
            else:
                continue


        title = "Raster plot of spike events"
        if param_dict is not None:
            title += "\n"
            for key, value in param_dict.items():
                title += f"{key}: {value} "
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Electrode")
        plt.legend()
        plt.show()

    def plot_spike_count_per_stim(self):
        """Plots the spike count per minute for each channel."""
        spike_count_df = self._db.get_spike_count(
            self.start_time, self.stop_time, self.fs_experiment.exp_name
        )
        # spike_count_df = spike_count_df[spike_count_df["electrode"].isin(ELECTRODES)]
        plt.figure(figsize=(80, 20))
        sns.lineplot(
            data=spike_count_df,
            x="Time",
            y="Spike per minutes",
            hue="channel",
            palette="colorblind",
        )
        plt.yscale("log")
        plt.title("Spike count per minute")
        plt.xlabel("Time")
        plt.ylabel("Spike count")

        for time_, stim in self._stim_history.items():
            plt.axvline(time_, color="black", linestyle="--", alpha=0.5)
            plt.text(time_, spike_count_df["Spike per minutes"].max() + 1, f"{stim}")
        plt.show()

    def plot_all_stims_for_channel(
        self,
        channel,
        s_before=1,
        s_after=2.5,
        show_electrodes=None,
        guideline_freq=None,
    ):
        """Plots the raster plot of all stimulations for a given channel.

        Args:
        channel: int
            The channel to plot the raster for.
        s_before: float
            The time before the first stimulation for which to plot in seconds. Note that if the value is large you may see previous stimulations on other channels.
        s_after: float
            The time after the last stimulation for which to plot in seconds. Note that if the value is large you may see later stimulations on other channels.
        show_electrodes: list[int]
            The list of electrodes to show in the plot. If None, all electrodes are shown.
        guideline_freq: str
            The frequency at which to show vertical lines to guide the eye. Will be aligned to the first stimulation time and shown before, during and after the stimulations.
        """
        stim_df = self.get_stimulation_parameter_history()
        stim_df = stim_df[stim_df["channel"].apply(lambda x: channel in x)]
        for param_id in stim_df["param_id"].unique():
            try:
                param_df = stim_df[stim_df["param_id"] == param_id]
                self._plot_raster_for_channels(
                    param_df,
                    show_electrodes=show_electrodes,
                    s_before=s_before,
                    s_after=s_after,
                    param_dict=self.parameters[param_id].get_names(),
                    guideline_freq=guideline_freq,
                )
            except ValueError:
                print(f"No events for parameter {param_id}.")
                continue

    def plot_all_stims_for_param(
        self,
        param_id,
        s_before=1,
        s_after=2.5,
        show_electrodes=None,
        guideline_freq=None,
    ):
        """Plots the raster plot of all stimulations for a given parameter.

        Args:
        param_id: int
            The parameter ID to plot the raster for. See the self.get_stimulation_parameter_history() method to get the parameter IDs from the dataframe.
        s_before: float
            The time before the first stimulation for which to plot in seconds. Note that if the value is large you may see previous stimulations on other channels.
        s_after: float
            The time after the last stimulation for which to plot in seconds. Note that if the value is large you may see later stimulations on other channels.
        show_electrodes: list[int]
            The list of electrodes to show in the plot. If None, all electrodes are shown.
        guideline_freq: str
            The frequency at which to show vertical lines to guide the eye. Will be aligned to the first stimulation time and shown before, during and after the stimulations.
        """
        stim_df = self.get_stimulation_parameter_history()
        stim_df = stim_df[stim_df["param_id"] == param_id].copy()
        for stim_channel in self.scan_channels:
            param_df = stim_df[stim_df["channel"].apply(lambda x: stim_channel in x)]
            try:
                self._plot_raster_for_channels(
                    param_df,
                    show_electrodes=show_electrodes,
                    s_before=s_before,
                    s_after=s_after,
                    param_dict=self.parameters[param_id].get_names(),
                    guideline_freq=guideline_freq,
                )
            except ValueError:
                print(f"No events for channel {stim_channel} for parameter {param_id}.")
                continue

    def plot_all_stims(
        self,
        s_before=1,
        s_after=2.5,
        show_electrodes=None,
        guideline_freq=None,
    ):
        """Plots the raster plot of all stimulations.

        NOTE : May output a very large number of plots if the scan is large.

        Args:
        s_before: float
            The time before the first stimulation for which to plot in seconds. Note that if the value is large you may see previous stimulations on other channels.
        s_after: float
            The time after the last stimulation for which to plot in seconds. Note that if the value is large you may see later stimulations on other channels.
        show_electrodes: list[int]
            The list of electrodes to show in the plot. If None, all electrodes are shown.
        guideline_freq: str
            The frequency at which to show vertical lines to guide the eye. Will be aligned to the first stimulation time and shown before, during and after the stimulations
        """
        for channel in self.scan_channels:
            self.plot_all_stims_for_channel(
                channel,
                s_before=s_before,
                s_after=s_after,
                show_electrodes=show_electrodes,
                guideline_freq=guideline_freq,
            )
