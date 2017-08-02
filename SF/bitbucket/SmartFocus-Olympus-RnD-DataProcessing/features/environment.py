import ipdb
import shutil
import os
# import re

BEHAVE_DEBUG_ON_ERROR = False

def setup_debug_on_error(userdata):
    global BEHAVE_DEBUG_ON_ERROR
    BEHAVE_DEBUG_ON_ERROR = userdata.getbool("BEHAVE_DEBUG_ON_ERROR")

def before_all(context):
    rootpath = "/Users/panagiotis.agisoikonomou-filandras/Workspace/smartfocus/python_projects/processing/data"
    setup_debug_on_error(context.config.userdata)
    context.rootpath = rootpath


def after_step(context, step):
    if BEHAVE_DEBUG_ON_ERROR and step.status == "failed":
        # -- ENTER DEBUGGER: Zoom in on failure location.
        # NOTE: Use IPython debugger, same for pdb (basic python debugger).
        ipdb.post_mortem(step.exc_traceback)
