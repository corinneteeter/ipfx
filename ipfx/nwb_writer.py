import pynwb
from allensdk.core.nwb_data_set import NwbDataSet
import ipfx.nwb_reader as nwb_reader


class NwbWriter(object):

    def __init__(self, nwb_file_name):
        self.nwb_file_name = nwb_file_name

    def add_spike_times(self, sweep_num, spike_times):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class Nwb1Writer(NwbWriter):

    def __init__(self, nwb_file_name):
        NwbWriter.__init__(self, nwb_file_name)
        self.nwbfile = NwbDataSet(self.nwb_file_name)

    def add_spike_times(self, sweep_num, spike_times):
        self.nwbfile.set_spike_times(sweep_num, spike_times)

    def save(self):
        pass


class Nwb2Writer(NwbWriter):

    def __init__(self, nwb_file_name):
        NwbWriter.__init__(self, nwb_file_name)

        io = pynwb.NWBHDF5IO(self.nwb_file_name, 'a')
        self.nwbfile = io.read()
        io.close()

        self.nwbfile.add_unit_column('sweep_number', description="sweep number")

    def add_spike_times(self, sweep_num, spike_times):
        self.nwbfile.add_unit(spike_times=spike_times, sweep_number=sweep_num)

    def save(self):
        io = pynwb.NWBHDF5IO(self.nwb_file_name, 'w')
        io.write(self.nwbfile)
        io.close()


def create_nwb_writer(nwb_file):
    """Create an appropriate writer of the nwb_file

    Parameters
    ----------
    nwb_file: str file name

    Returns
    -------
    writer object
    """

    nwb_version = nwb_reader.get_nwb_version(nwb_file)

    if nwb_version["major"] == 2:
        return Nwb2Writer(nwb_file)
    elif nwb_version["major"] == 1 or nwb_version["major"] == 0:
        return Nwb1Writer(nwb_file)
    else:
        raise ValueError("Unsupported or unknown NWB major" +
                         "version {} ({})".format(nwb_version["major"], nwb_version["full"]))
