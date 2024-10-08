import jsonpickle
import json
import re
import pathlib
import subprocess
import uuid
import pprint
import re
import ast
import base64
import pickle
import os
import glob
import sys
import argparse

from collections import defaultdict
from toposort import toposort, toposort_flatten
from pathlib import Path

import xml.etree.ElementTree as et

def cmd_no_output(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        #output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        return output
    except subprocess.CalledProcessError as e:
        # I think this is mostly safe, since something like the OOM killer
        # will send signal SIGKILL rather than SIGTERM. But I need
        # to verify this.
        print(f"RETURN CODE {e.returncode}, ERROR {e.output}")
        if e.returncode == 143:
            print(f"Command {cmd} received SIGTERM. "\
                "This is probably the result of 'trap \"kill 0\" EXIT\"' "\
                "in the invoked command, so it is being ignored.")
            return ""
        else:
            print(f"Command '{cmd}' failed with return code {e.returncode} "\
                "and error {e.output}")
            exit(1)
    #subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    #subprocess.call(cmd, shell=True)


def run_sif_cmd(sif_path, cmd, env_str="", show_cmd=False):
    # Writable TMPFS flag is needed because STAR align creates symlink,
    # which is impossible on read-only file system (default).
    # This needs a more robust fix.
    sif_cmd = f"sudo {env_str} singularity exec -i --writable-tmpfs --pwd / --cleanenv -B .:/data {sif_path} sh -c \"{cmd}\""
    if show_cmd:
        print(f"Executing '{sif_cmd}'")
    return cmd_no_output(sif_cmd)

def get_docker_entrypoint(job_dir, img, sif):
    ept_file_name = os.path.join(job_dir, sif + "_ept")
    img_info_json = cmd_no_output(f"docker inspect {img}")
    img_info = json.loads(img_info_json)
    ept_arr = img_info[0]["Config"]["Entrypoint"]
    ept_str = " ".join(ept_arr) if ept_arr is not None else ""

    with open(ept_file_name, "w+") as f:
        f.write(ept_str)
    return  ept_str 

def get_sif_entrypoint(job_dir, img):
    ept_file_name = os.path.join(job_dir, img + "_ept")
    if os.path.isfile(ept_file_name):
        with open(ept_file_name, "r") as f:
            return f.read()

    print(f"File {ept_file_name} does not exist.")
    exit(1)

    

def build_singularity_img(job_dir):
    uuid1 = str(uuid.uuid4())
    uuid2 = str(uuid.uuid4())
    docker_dir = os.path.join(job_dir, "Dockerfiles")
    
    # Don't rebuild if docker dir already has image in it.
    # This needs to be refactored eventually to make sure
    # sif contents align with those of dockerfile in case
    # of updates to latter.
    for sif in pathlib.Path(docker_dir).glob("*.sif"):
        img_name = Path(sif).stem
        ept = get_sif_entrypoint(docker_dir, img_name)
        return (sif, ept)

    cmd_no_output(f"docker build -t {uuid1} {docker_dir}")
    cmd_no_output(f"docker save {uuid1} -o {uuid2}.tar")
    entrypoint = get_docker_entrypoint(docker_dir, uuid1, uuid2)
    sif_path = os.path.join(docker_dir, f"{uuid2}.sif")
    cmd_no_output(f"singularity build {sif_path} docker-archive://{uuid2}.tar")
    run_sif_cmd(sif_path, "mkdir -p /tmp/output")
    cmd_no_output(f"docker rmi {uuid1}")
    cmd_no_output(f"rm {uuid2}.tar")
    
    return (sif_path, entrypoint)


def parse_pattern_query(pq):
    path = pq["root"]
    pattern = pq["pattern"]

    data_prefix = False
    if path.startswith("/data/"):
        data_prefix = True
        path = "./" + path[6:]
    if pattern.startswith("**/"):
        pattern = pattern[3:]

    cmd = f"sudo find {path} -name {pattern} "
    matches = []

    if pq["findFile"]:
        out = cmd_no_output(cmd + "-type f")
        matches.extend(out.split("\n"))
    if pq["findDir"]:
        out = cmd_no_output(cmd + "-type d")
        matches.extend(out.split("\n"))
    print(f"Assignging pq {pq} as {matches}")
    if len(matches):
        res = matches[0]
        if data_prefix:
            return "/data/" + matches[0][2:]
        return matches[0]
    return None
        
def param_str(k, v, widget_conf):
    
    # This is a hack to deal with the fact that downloadURL's
    # props don't give an entry for "skipDownload", but I need
    # a more robust solution in long term.
    if k not in widget_conf['parameters']:
        return ''

    print("TYPE")
    print(widget_conf['parameters'][k]['type'])
    if widget_conf['parameters'][k]['type'] == 'patternQuery':
        v = parse_pattern_query(v)
        

    pstr = ''
    conf = widget_conf['parameters'][k]
    if conf['flag'] is None and ('argument' not in conf or not conf['argument']):
        return ''
    if v and conf['flag'] is not None:
        pstr += conf['flag'] + ' '
    if v and isinstance(v, str):
        pstr += v + ' '
    if v and isinstance(v, list):
        pstr += ' '.join(v) + ' '

    # Python bools are technically ints
    if v and isinstance(v, int) and not isinstance(v, bool):
        pstr += str(v) + ' '
    return pstr

def get_env_str(k, v, widget_conf):
    if k not in widget_conf['parameters']:
        return '' 
    if widget_conf['parameters'][k]['type'] == 'patternQuery':
        v = parse_pattern_query(v)
    conf = widget_conf['parameters'][k]
    estr = ''
    if 'env' not in conf or conf['env'] is None:
        return ''
    if v and conf['env'] is not None:
        estr += "SINGULARITYENV_" + conf['env'] + '='
    if v and isinstance(v, str):
        estr += '"' + v + '" '
    if v and isinstance(v, list):
        estr += '['
        for (ind, el) in enumerate(v):
            estr += '"' + el + '"'
            estr += "," if el != len(v) else ""
        estr += ']'

    # Python bools are technically ints
    if v and isinstance(v, int) and not isinstance(v, bool):
        estr += str(v) + ' '
    return estr

def cmd_substitution(cmd, props, inputs):
    def replace_var(match):
        var_name = match.group(1)
        if var_name in inputs:
            return inputs[var_name]
        if var_name in props:
            return props[var_name]
        print(f"Cmd {cmd} references undefined var {var_name}")
        exit(1)

    var_pattern = r"_bwb\{(.*?)\}"
    out = re.sub(var_pattern, replace_var, cmd)
    out = re.sub("\$", "\$", out)
    out = re.sub("\"", "\\\"", out)
    return out

def get_job_cmd(widget_dir, props, inputs, entrypoint):
    pprint.pp(props)
    widget_name = os.path.basename(widget_dir)
    widget_json_path = os.path.join(widget_dir, f"{widget_name}.json")
    print(f"{widget_name} props:")
    with open(widget_json_path, 'r') as f:
        widget_json_raw = f.read()
    widget_conf = jsonpickle.decode(widget_json_raw)

    #command = widget_conf['command'][0] + ' '
    command = entrypoint + " "
    parsed_cmds = map(lambda x: cmd_substitution(x, props, inputs), 
        widget_conf['command'])
    command += ' && '.join(parsed_cmds) + ' '
    env_str = ''

    # Way to prevent downloadUrl widget from redownloading massive
    # fastq files. Remove this later.
    if widget_name == "downloadURL":
        command += ' --noClobber '

    required_params = widget_conf['requiredParameters']

    for k in required_params:
        if k in inputs:
            print(f"REQ INPS: Adding {k} = {inputs[k]}")
            env_str += get_env_str(k, inputs[k], widget_conf)
            command += param_str(k, inputs[k], widget_conf)
        else:
            print(f"REQ PROPS: Adding {k} = {props[k]}")
            env_str += get_env_str(k, props[k], widget_conf)
            command += param_str(k, props[k], widget_conf)
            

    for k in inputs:
        print(f"INPUTS: Adding {k} = {inputs[k]}")
        if k not in set(required_params):
            env_str += get_env_str(k, props[k] if k in props else inputs[k], widget_conf)
            command += param_str(k, inputs[k], widget_conf)
    
    params = widget_conf['parameters']
    for k in params:
        if k in required_params or k in inputs:
            continue

        #if k in props:
        #    print(f"Adding props param {k} = {param_str(k, props[k], widget_conf)}")
        #    command += param_str(k, props[k], widget_conf)
        #    env_str += get_env_str(k, props[k], widget_conf)

        elif (k in props['optionsChecked'] and props['optionsChecked'][k]
            or k == "decompress"):
            if k in props:
                print(f"Adding {k} = {props[k]}")
                command += param_str(k, props[k], widget_conf)
                env_str += get_env_str(k, props[k], widget_conf)
            elif 'default' in params['k'] and params['k']['default']:
                print(f"Adding {k} = {params[k]['default']}")
                command += param_str(k, params[k]['default'], widget_conf)
                env_str += get_env_str(k, params[k]['default'], widget_conf)
                

        elif 'default' in params[k] and params[k]['default']:
            print(f"Testing param {k}, {props['optionsChecked'][k]}")
            # For some reason, decompress is listed as false in optionsChecked,
            # even though the GUI shows it as checked?
            if k != "decompress" and not props['optionsChecked'][k]:
                continue
            # print(f"Adding default param {k} = {param_str(k, params[k]['default'], widget_conf)}")
            command += param_str(k, params[k]['default'], widget_conf)
            env_str += get_env_str(k, params[k]['default'], widget_conf)

    return env_str, command


def get_sif_outputs(sif_path):
    outputs = {}
    ls_output = run_sif_cmd(sif_path, "ls /tmp/output")
    outfiles = ls_output.strip().split('\n')
    for outfile in outfiles:
        outfile_val = run_sif_cmd(sif_path, f"cat /tmp/output/{outfile}")
        outputs[outfile] = outfile_val
    return outputs
                    
    
def parse_ows(ows_path):
    links = defaultdict(lambda: defaultdict(list))
    ids_to_names = {}
    dependencies = defaultdict(set)
    properties = {}

    with open(ows_path, 'r') as f:
        ows_raw = f.read()
    
    root = et.fromstring(ows_raw)
    for node in root.find("nodes"):
        node_id = node.get("id")
        node_name = node.get("name")
        ids_to_names[node_id] = node_name

    for prop_set in root.find("node_properties"):
        node_id = prop_set.get("node_id")
        fmt = prop_set.get("format")
        raw = prop_set.text

        if fmt == "literal":
            # savedWidgetGeometry field has weird byte str stuff that AST won't accept.
            raw_cleaned = re.sub(r"'savedWidgetGeometry':.*?(?=, |}),", "", raw)
            properties[node_id] = ast.literal_eval(raw_cleaned)

        if fmt == "pickle":
            decoded_bytes = base64.b64decode(raw)
            properties[node_id] = pickle.loads(decoded_bytes)
        node_id = node.get("id")
        node_name = node.get("name")
        ids_to_names[node_id] = node_name

    for link in root.find("links"):
        link_src = link.get("source_node_id")
        link_dst = link.get("sink_node_id")
        link_inp = link.get("source_channel")
        link_out = link.get("sink_channel")

        dependencies[link_dst].add(link_src)
        links[link_src][link_dst].append( (link_inp, link_out) )
    
    return {
        "ids_to_names": ids_to_names,
        "top_sort": toposort_flatten(dependencies),
        "links": links,
        "properties": properties
    }

def update_job_inputs(job_id, outlinks, src_outputs, props_dict, inputs_dict):
    for dst_node, outputs in outlinks.items():
        for output in outputs:
            src_name, dst_name = output

            # Noticing some odd behavior here. Inputs which are
            # specified via links in "OWS" file do not exist as
            # output files in /tmp/output of source node contai
            # ner.
            if src_name in src_outputs:
                inputs_dict[dst_node][dst_name] = ' '.join(src_outputs[src_name].split("\n"))
                # print(f"OUTPUTS: Forwarding {src_name} ({dst_name}) from {job_id} to {dst_node} w/ value {src_outputs[src_name]}")

            elif src_name in inputs_dict[job_id]:
                inputs_dict[dst_node][dst_name] = inputs_dict[job_id][src_name]
                # print(f"INPUTS: Forwarding {src_name} ({dst_name}) from {job_id} to {dst_node} w/ value {inputs_dict[job_id][src_name]}")

            # NEW: If link references prop of source job, pass that along,
            # even if it is not given in source job /tmp/output.
            elif src_name in props_dict[job_id]:
                inputs_dict[dst_node][dst_name] = props_dict[job_id][src_name]
                # print(f"PROPS: Forwarding {src_name} ({dst_name}) from {job_id} to {dst_node} w/ value {props_dict[job_id][src_name]}")
    

def gen_workflow_command(workflow_dir, user_inputs={}):
    widgets_dir = os.path.join(workflow_dir, "widgets")
    widget_child_dir = [ f for f in os.scandir(widgets_dir) if f.is_dir() ]
    widget_dir = widget_child_dir[0]

    workflow_name = os.path.basename(workflow_dir)
    workflow_ows_path = os.path.join(workflow_dir, f"{workflow_name}.ows")
    ows_ret = parse_ows(workflow_ows_path)

    job_id_to_name = ows_ret["ids_to_names"]
    job_io_links = ows_ret["links"]
    job_props = ows_ret["properties"]
    job_top_sort = ows_ret["top_sort"]

    #print("TS")
    #for el in job_top_sort:
    #    print(f"\t{job_id_to_name[el]}, ID {el}")
    #exit(0)
    
    # I've selected an ordering of jobs for salmon_demo
    # to help speed up testing. Be sure to remove this later.
    #job_top_sort = ['4', '0', '5', '3', '1', '2', '6', '7', '8', '9', '10', '11', '12']

    inputs = {}
    for job_id in job_top_sort:
        inputs[job_id] = {}
    inputs[job_top_sort[0]] = user_inputs

    for job_id in job_top_sort:
        print(f"Executing job {job_id_to_name[job_id]} (ID {job_id})")
        job_name = job_id_to_name[job_id]
        job_dir = os.path.join(widget_dir, job_name)

        (job_img_path, entrypoint) = build_singularity_img(job_dir)
        env_str, job_cmd = get_job_cmd(job_dir, job_props[job_id], inputs[job_id], entrypoint)
        run_sif_cmd(job_img_path, job_cmd, env_str, show_cmd=True)
        outputs = get_sif_outputs(job_img_path)
        update_job_inputs(job_id, job_io_links[job_id], outputs, job_props, inputs)
        print(f"Job {job_id_to_name[job_id]} (ID {job_id}) completed successfully\n\n")

    widgets_dir = os.path.join(workflow_dir, "widgets")
    widget_child_dir = [ f for f in os.scandir(widgets_dir) if f.is_dir() ]
    widget_dir = widget_child_dir[0]


def get_input_assignments(inputs):
    if inputs is None:
        return {}

    inp_pairs = {}
    for inp in inputs:
        split_str = inp.split("=")
        if len(split_str) != 2:
            print(f"ERR: Received invalid input {inp}."\
                "Inputs must be of form NAME=VAL")
            exit(1)
        inp_pairs[split_str[0]] = split_str[1]
    return inp_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_workflow.py",
        description="Run a BWB workflow with singularity",
        epilog="")
    parser.add_argument("workflow_dir")
    parser.add_argument("-i", "--input", required=False, nargs='*',
        help="Assign value to start node's parameter with form NAME=VAL")
    args = parser.parse_args()

    inp_pairs = get_input_assignments(args.input)
    gen_workflow_command(args.workflow_dir, inp_pairs)
