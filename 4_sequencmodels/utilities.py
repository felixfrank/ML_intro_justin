import pretty_midi
import tensorflow as tf


def load_song(filename):
    pm = pretty_midi.PrettyMIDI(filename)
    return pm.fluidsynth()


def _float32_feature(value):
    return tf.train.Feature(tf.train.FloatList(value=value))


def create_tf_record_from_midi(savefile, files):
    writer = tf.python_io.TFRecordWriter(savefile)
    data = [load_song(file) for file in files[:2]]

    for d in data:
        feats = tf.train.Features(feature={'midi': tf.train.Feature(float_list=tf.train.FloatList(value=d))})
        example = tf.train.Example(features=feats)
        writer.write(example.SerializeToString())

    writer.close()


def extract_piano_roll(filename):
    from music21 import converter, instrument, note, chord
    notes = []
    midi = converter.parse(filename)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts: # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


def data_to_onehot(data):
    import numpy as np
    out = []
    for sequence in data:
        seqlen = len(sequence)
        out_seq = np.zeros((seqlen, 108-20), dtype=np.int8)
        for i in range(seqlen):
            idx = np.array(sequence[i]) - 21
            if idx.size > 0:
                out_seq[i, idx] = 1
        out.append(out_seq)

    return out

def data_to_dataset(data):
    import numpy as np
    batchsize = len(data)
    seqlen = np.max([len(seq) for seq in data])
    obs_size = 108-20
    out = np.zeros((batchsize, seqlen, obs_size))
    mask = np.zeros((batchsize, seqlen), dtype=np.bool)
    i = 0
    for sequence in data:
        seqlen = len(sequence)
        mask[i, :seqlen] = 1
        for ii in range(seqlen):
            idx = np.array(sequence[ii]) - 21
            if idx.size > 0:
                out[i, ii, idx] = 1
        i += 1

    return out, mask