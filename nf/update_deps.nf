process update_deps {
  label 'local_job'
  output:
  stdout

  """
  pwd
  """  
}
workflow {
  update_deps | view
}
