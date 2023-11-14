from sbi import inference


def smcabc(loss, prior, num_workers=-1, **kwargs):
    """
    Performs SMCABC inference using the SBI pckage.
    """

    def distance(y, x):
        return x.reshape(-1)

    simulator, prior = inference.prepare_for_sbi(loss, prior)
    smcabc_sampler = inference.SMCABC(
        simulator, prior, num_workers=num_workers, distance=distance
    )
    smcabc_sampler.distance = distance  # bug in SBI?
    samples, summary = smcabc_sampler(0.0, **kwargs)
    return samples, summary
