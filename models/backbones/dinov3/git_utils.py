import git


git_repo = git.Repo(__file__, search_parent_directories=True)
GIT_ROOT = git_repo.git.rev_parse('--show-toplevel')
