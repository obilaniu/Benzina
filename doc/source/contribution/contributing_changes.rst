Contributing changes
====================

Licensing of contributed material
---------------------------------

Keep in mind as you contribute, that code, docs and other material submitted to
open source projects are usually considered licensed under the same terms
as the rest of the work.

The details vary from project to project, but from the perspective of this
document's authors:

- Anything submitted to a project falls under the licensing terms in the
  repository's top level ``LICENSE`` file.

    - For example, if a project's ``LICENSE`` is BSD-based, contributors should
      be comfortable with their work potentially being distributed in binary
      form without the original source code.

- Per-file copyright/license headers are typically extraneous and undesirable.
  Please don't add your own copyright headers to new files unless the project's
  license actually requires them!

    - Not least because even a new file created by one individual (who often
      feels compelled to put their personal copyright notice at the top) will
      inherently end up contributed to by dozens of others over time, making a
      per-file header outdated/misleading.

Version control branching
-------------------------

* Always **make a new branch** for your work, no matter how small. This makes
  it easy for others to take just that one set of changes from your repository,
  in case you have multiple unrelated changes floating around.

    * A corollary: **don't submit unrelated changes in the same branch/pull
      request**! The maintainer shouldn't have to reject your awesome bugfix
      because the feature you put in with it needs more review.

* **Base your new branch off of the appropriate branch** on the main
  repository:

    * **Bug fixes** should be based on the branch named after the **oldest
      supported release line** the bug affects.

        * E.g. if a feature was introduced in 1.1, the latest release line is
          1.3, and a bug is found in that feature - make your branch based on
          1.1.  The maintainer will then forward-port it to 1.3 and master.
        * Bug fixes requiring large changes to the code or which have a chance
          of being otherwise disruptive, may need to base off of **master**
          instead. This is a judgement call -- ask the devs!

    * **New features** should branch off of **the 'master' branch**.

        * Note that depending on how long it takes for the dev team to merge
          your patch, the copy of ``master`` you worked off of may get out of
          date! If you find yourself 'bumping' a pull request that's been
          sidelined for a while, **make sure you rebase or merge to latest
          master** to ensure a speedier resolution.

Code formatting
---------------

* **Follow the style you see used in the primary repository**! Consistency with
  the rest of the project always trumps other considerations. It doesn't matter
  if you have your own style or if the rest of the code breaks with the greater
  community - just follow along.
* Python projects usually follow the `PEP-8
  <http://www.python.org/dev/peps/pep-0008/>`_ guidelines (though many have
  minor deviations depending on the lead maintainers' preferences.)

Documentation isn't optional
----------------------------

It's not! Patches without documentation will be returned to sender.  By
"documentation" we mean:

* **Docstrings** (for Python; or API-doc-friendly comments for other languages)
  must be created or updated for public API functions/methods/etc. (This step
  is optional for some bugfixes.)

    * Don't forget to include `versionadded
      <http://sphinx-doc.org/markup/para.html#directive-versionadded>`_/`versionchanged
      <http://sphinx-doc.org/markup/para.html#directive-versionchanged>`_ ReST
      directives at the bottom of any new or changed Python docstrings!

        * Use ``versionadded`` for truly new API members -- new methods,
          functions, classes or modules.
        * Use ``versionchanged`` when adding/removing new function/method
          arguments, or whenever behavior changes.

* New features should ideally include updates to **prose documentation**,
  including useful example code snippets.
* All submissions should have a **changelog entry** crediting the contributor
  and/or any individuals instrumental in identifying the problem.

Full example
------------

Here's an example workflow for the project ``Benzina``, which
is currently in hypothetic version 1.0.x. Your username is ``yourname`` and you're
submitting a basic bugfix.

Preparing your Fork
^^^^^^^^^^^^^^^^^^^

1. Click 'Fork' on Github, creating e.g. ``yourname/Benzina``.
2. Clone your project: ``git clone git@github.com:yourname/Benzina``.
3. ``cd Benzina``
4. `Create and activate a virtual environment <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_.
5. Install the development requirements: ``pip install -r dev-requirements.txt``.
6. Create a branch: ``git checkout -b foo-the-bars 1.0``.

Making your Changes
^^^^^^^^^^^^^^^^^^^

1. Add changelog entry crediting yourself.
2. Hack, hack, hack.
3. Commit your changes: ``git commit -m "Foo the bars"``

Creating Pull Requests
^^^^^^^^^^^^^^^^^^^^^^

1. Push your commit to get it back up to your fork: ``git push origin HEAD``
2. Visit Github, click handy "Pull request" button that it will make upon
   noticing your new branch.
3. In the description field, write down issue number (if submitting code fixing
   an existing issue) or describe the issue + your fix (if submitting a wholly
   new bugfix).
4. Hit 'submit'! And please be patient - the maintainers will get to you when
   they can.
