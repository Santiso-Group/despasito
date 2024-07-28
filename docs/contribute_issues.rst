
.. _issue:

Have an Issue? Create an Issue or Discussion
===============================================

If your questions and issues are not addressed in our :ref:`faqs` page or illustrated in our ``despasito.examples`` directory, you've come to the right place.

Before Anything Else
----------------------------
Please work your way through this checklist:

#. **Is the issue within despasito?**
    - If you have a reoccurring error with your input JSON file, there's not much we can do. You might write your input file as a dictionary in python and then write it to a file with ``json.dump(some_dictionary, open(filename, "w"), indent=4)``.
    - Error with a supporting python package, try to resolve on your own and check that your version of python is within limits. If the compatibility between a supporting package and a version of python we claim is/isn't supported by DESPASITO is a problem, create an issue.
#. **Are all of your parameters in the correct units?** Some errors result from parameters that are technically "in the realm of possibility" but aren't actually feasible. For example, the allowed range of values for the size parameter in saft might have a lower bound of zero, but a value of the order of 1e-5 does not provide meaningful results. (If the units are not provided in the documentation please create an issue.)
#. **Are you receiving a meaningful error message?** Some errors may occur from "outrageous" parameters that cannot achieve the calculation you think they can.
#. **Check previous and open** `issues <https://github.com/Santiso-Group/despasito/issues>`_
#. **Check previous and open** `discussions <https://github.com/Santiso-Group/despasito/discussions>`_

What Does Creating a New 'Issue' or 'Discussion' Entail?
---------------------------------------------------------
Ok, you've gone through the checklist above and still haven't found a solution to your problem, time to make an issue or start a discussion. 
Whether you're interested in assistance with a new feature, a mysterious error is not clear/meaningful, or even if there's a typo in the documentation, please make a new issue or discussion, but which one?

In either case, add one of the following prefixes to a descriptive title: *EOS, Thermo, Fitting, General, Docs, Tests*, or *Import*.

 * For *Bugs* make an `issue <https://github.com/Santiso-Group/despasito/issues>`_.
 * For *Ideas* and *Questions* start a `discussion <https://github.com/Santiso-Group/despasito/discussions>`_.

**Thanks for helping us improve DESPASITO!**


