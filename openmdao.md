# Python/OpenMDAO Install Notes

* Install Julia
* Create and activate a Python virtual environment
* `pip install julia`
* Open up a python prompt and do this:

  ```
  >>> import julia
  >>> julia.install()
  ```

  That gave a Julia error indicating that `PyCall` wasn't installed, and then
  installed it for me.

* Next check out `CCBlade.jl` from GitHub and install it:

  ```
  pkg> develop <path/to/CCBlade.jl>
  ```

Now the Python scripts should work.
