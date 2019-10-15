import pytest
import datetime
import pynwb
import h5py

from ipfx.x_to_nwb.NWBConverter import NWBConverter
from .helpers_for_tests import diff_h5, validate_nwb
from ipfx.bin.run_nwb1_to_nwb2_conversion import make_nwb2_file_name
from hdmf import Container
import numpy as np
import ipfx.nwb_reader as nwb_reader

from ipfx.bin.run_feature_extraction import embed_spike_times





def make_skeleton_nwb1_file(nwb1_file_name):

    with h5py.File(nwb1_file_name, 'w') as fh:
        dt = h5py.special_dtype(vlen=bytes)
        dset = fh.create_dataset("nwb_version", (1,), dtype=dt)
        dset[:] = "NWB-1"
        fh.create_group("acquisition/timeseries")
        fh.create_group("analysis")


def make_skeleton_nwb2_file(nwb2_file_name):

    nwbfile = pynwb.NWBFile(
        session_description='test icephys',
        identifier='session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    device = nwbfile.create_device(name='electrode_0')
    electrode = nwbfile.create_ic_electrode(name="elec0",
                                            description='intracellular electrode',
                                            device=device)

    io = pynwb.NWBHDF5IO(nwb2_file_name, 'w')
    io.write(nwbfile)
    io.close()



def test_embed_spike_times_into_nwb2(tmpdir_factory):

    sweep_spike_times = {
        3: [56.0, 44.6, 661.1],
        4: [156.0, 144.6, 61.1]
    }

    input_nwb_file_name = str(tmpdir_factory.mktemp("embed_spikes").join("input_v2.nwb"))
    output_nwb_file_name = str(tmpdir_factory.mktemp("embed_spikes").join("output_v2.nwb"))

    make_skeleton_nwb2_file(input_nwb_file_name)

    embed_spike_times(input_nwb_file_name, output_nwb_file_name, sweep_spike_times)

    nwb_data = nwb_reader.create_nwb_reader(output_nwb_file_name)

    for sweep_num, spike_times in sweep_spike_times.items():
        assert np.allclose(nwb_data.get_spike_times(sweep_num), spike_times)


def test_embed_spike_times_into_nwb1(tmpdir_factory):

    sweep_spike_times = {
        3: [56.0, 44.6, 661.1],
        4: [156.0, 144.6, 61.1]
    }

    input_nwb_file_name = str(tmpdir_factory.mktemp("embed_spikes").join("input_v1.nwb"))
    output_nwb_file_name = str(tmpdir_factory.mktemp("embed_spikes").join("output_v1.nwb"))

    make_skeleton_nwb1_file(input_nwb_file_name)
    embed_spike_times(input_nwb_file_name, output_nwb_file_name, sweep_spike_times)

    nwb_data = nwb_reader.create_nwb_reader(output_nwb_file_name)

    for sweep_num, spike_times in sweep_spike_times.items():
        assert np.allclose(nwb_data.get_spike_times(sweep_num), spike_times)


class ForTestEmbedSpikeTimesNwb(object):

    def make_skeleton_nwb_file(self,nwb_file_name):
        raise NotImplementedError

    def test_embed_spike_times_into_nwb(self, tmpdir_factory):
        sweep_spike_times = {
            3: [56.0, 44.6, 661.1],
            4: [156.0, 144.6, 61.1]
        }
        tmp_dir = tmpdir_factory.mktemp("embed_spikes_into_nwb")
        input_nwb_file_name = str(tmp_dir.join("input.nwb"))
        output_nwb_file_name = str(tmp_dir.join("output.nwb"))

        self.make_skeleton_nwb_file(input_nwb_file_name)
        embed_spike_times(input_nwb_file_name, output_nwb_file_name, sweep_spike_times)

        nwb_data = nwb_reader.create_nwb_reader(output_nwb_file_name)

        for sweep_num, spike_times in sweep_spike_times.items():
            assert np.allclose(nwb_data.get_spike_times(sweep_num), spike_times)


class TestEmbedSpikeTimesNwb1(ForTestEmbedSpikeTimesNwb):

    def make_skeleton_nwb_file(self, nwb_file_name):
        with h5py.File(nwb_file_name, 'w') as fh:
            dt = h5py.special_dtype(vlen=bytes)
            dset = fh.create_dataset("nwb_version", (1,), dtype=dt)
            dset[:] = "NWB-1"
            fh.create_group("acquisition/timeseries")
            fh.create_group("analysis")


class TestEmbedSpikeTimesNwb2(ForTestEmbedSpikeTimesNwb):

    def make_skeleton_nwb_file(self, nwb_file_name):
        nwbfile = pynwb.NWBFile(
            session_description='test icephys',
            identifier='session_uuid',
            session_start_time=datetime.datetime.now(),
            file_create_date=datetime.datetime.now()
        )

        device = nwbfile.create_device(name='electrode_0')
        electrode = nwbfile.create_ic_electrode(name="elec0",
                                                description='intracellular electrode',
                                                device=device)

        io = pynwb.NWBHDF5IO(nwb_file_name, 'w')
        io.write(nwbfile)
        io.close()

