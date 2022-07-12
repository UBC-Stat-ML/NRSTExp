process update_deps {
  label 'local_job'

  """
  julia --project -e "using Pkg; Pkg.update()"
  """  
}
workflow {
  update_deps
}
