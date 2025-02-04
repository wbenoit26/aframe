import logging
import time

import numpy as np
from hermes.aeriel.client import InferenceClient
from tqdm import tqdm

from infer.data import Sequence
from infer.postprocess import Postprocessor


def infer(
    client: InferenceClient,
    sequence: Sequence,
    postprocessor: Postprocessor,
    return_raw_output: bool = False,
):
    """
    Perform inference on a sequence of data.

    Args:
        client:
            Inference client. Must already be connected to a Triton server.
        sequence:
            Sequence object
        postprocessor:
            Postprocessor object
        return_raw_output:
            Whether to return the raw (unclustered) inference outputs.
            Defaults to False.

    Returns:
        background: Background events
        foreground: Foreground events
        raw_background: Raw background outputs (if return_raw_output is True)
        raw_foreground: Raw foreground outputs (if return_raw_output is True)
    """
    logging.info(
        "Beginning inference on sequence {} corresponding "
        "to {}s of data from {} with shifts {} and sample rate {}, beginning "
        "at GPS time {}".format(
            sequence.id,
            sequence.duration,
            sequence.background_fname,
            sequence.shifts / sequence.sample_rate,
            sequence.sample_rate,
            sequence.t0,
        )
    )

    for i, (x, x_inj) in enumerate(tqdm(sequence)):
        sequence_start = i == 0
        sequence_end = i == len(sequence) - 1
        logging.debug(
            f"Submitting inference request {i} for sequence {sequence.id}"
        )
        client.infer(
            np.stack([x, x]),
            request_id=i,
            sequence_id=sequence.id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

        if x_inj is not None:
            # pass injected data __and__ background
            # data to be used for whitening
            client.infer(
                np.stack([x, x_inj]),
                request_id=i,
                sequence_id=sequence.id + 1,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )

        # wait for the first response to come back
        # for both sequences to allow for some
        # warm up and to check to be sure we're not
        # hitting any inference errors
        if not i:
            while not sequence.started:
                client.get()
                time.sleep(1e-2)

    result = client.get()
    while result is None:
        result = client.get()
        time.sleep(1e-1)
    logging.info("Inference complete, postprocessing output timeseries")

    raw_background, raw_foreground = result
    background = postprocessor(raw_background)
    foreground = postprocessor(raw_foreground)

    logging.info("Recovering injections from foreground events")
    foreground = sequence.recover(foreground)

    logging.info(f"Finished processing sequence {sequence.id}")
    if return_raw_output:
        return background, foreground, raw_background, raw_foreground
    return background, foreground
