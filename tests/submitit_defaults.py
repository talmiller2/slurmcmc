submitit_kwargs = {}
submitit_kwargs['slurm_partition'] = 'core'
# submitit_kwargs['slurm_constraint'] = 'serial'
submitit_kwargs['timeout_min'] = 10
submitit_kwargs['slurm_signal_delay_s'] = 30
submitit_kwargs['slurm_additional_parameters'] = {'no-requeue': True}
