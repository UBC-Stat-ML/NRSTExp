workflow {
  exps_ch = Channel.of('ess_versus_cost')
  mods_ch = Channel.of('MvNormal')
  cors_ch = Channel.of(0.99)
  updateDeps | runExp(exps_ch, mods_ch, cors_ch)
}

process updateDeps {
  label 'local_job'

  """
  julia --project -e "using Pkg; Pkg.update()"
  """  
}

process runExp {
  label 'parallel_job'
  publishDir 'BORRAR', mode: 'copy', overwrite: true
  input:
    val exper
    val model
    val maxcor
  output:
    path '*.csv'

  """
  julia --project -e "using NRSTExp; dispatch()" $exper $model $maxcor
  """  
}
