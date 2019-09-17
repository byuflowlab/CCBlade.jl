# Python/OpenMDAO Install Notes

* Install Julia
* Create and activate a Python virtual environment. Install OpenMDAO,
  `matplotlib`, and all the other Python-related stuff with `pip`.
* `$ pip install julia`
* Open up a python prompt and do this:

  ```
  >>> import julia
  >>> julia.install()
  ```

  That gave a Julia error indicating that `PyCall` wasn't installed, and then
  installed it for me.

* Next check out `CCBlade.jl` from GitHub and install it from the Julia REPL
  (type `]` in the Julia REPL to get to the package manager prompt):

  ```
  $ julia
  pkg> develop <path/to/CCBlade.jl>
  ```

Now the Python scripts in `src/` work (just do `python
<path/to/script.py>`) provided that it can find the airfoil data file.
