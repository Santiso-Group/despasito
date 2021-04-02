
.. _issue:

Have an Issue? Create an Issue
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
#. **Check previous and open** `issues <https://github.com/jaclark5/despasito/issues>`_
#. **Coming Soon to GitHub:** Check open *discussions*

What Does 'Create a New Issue' Entail?
---------------------------------------
Ok, you've gone through the checklist above and still haven't found a solution to your problem, time to make an issue. 
Whether you're interested in assistance with a new feature, a mysterious error is not clear/meaningful, or even if there's a typo in the documentation, please make a new issue. 

#. **Navigate to the** `issues <https://github.com/jaclark5/despasito/issues>`_ **page**
#. **Add one of the following prefixes:** EOS, Thermo, Fitting, General, Docs, Tests, Import
#. **Add a dash and one of the following terms:** Error, Edit, Feature, Idea
#. **End the prefix segment with a colon and proceed with a descriptive title.**
#. **In the body of the issue please provide the following information:**
    - A description of the issue, including the **full** error message (not just the last line).
    - Details on how to reproduce it (include input file code at the end of the text body).

**Thanks for helping us improve DESPASITO!**


