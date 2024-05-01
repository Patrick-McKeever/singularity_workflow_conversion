import jsonpickle
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

from collections import defaultdict
from toposort import toposort, toposort_flatten

import xml.etree.ElementTree as et

def cmd_no_output(cmd):
	print(f"Executing '{cmd}'")
	return subprocess.check_output(cmd, shell=True).decode('utf-8')
	#subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
	#subprocess.call(cmd, shell=True)

def run_sif_cmd(sif_path, cmd):
	return cmd_no_output(f"sudo singularity exec -i --pwd / --cleanenv -B .:/data {sif_path} {cmd}")

def build_singularity_img(job_dir):
	uuid1 = str(uuid.uuid4())
	uuid2 = str(uuid.uuid4())
	docker_dir = os.path.join(job_dir, "Dockerfiles")
	
	# Don't rebuild if docker dir already has image in it.
	# This needs to be refactored eventually to make sure
	# sif contents align with those of dockerfile in case
	# of updates to latter.
	for sif in pathlib.Path(docker_dir).glob("*.sif"):
		return sif

	cmd_no_output(f"docker build -t {uuid1} {docker_dir}")
	cmd_no_output(f"docker save {uuid1} -o {uuid2}.tar")
	cmd_no_output(f"docker rmi {uuid1}")
	sif_path = os.path.join(docker_dir, f"{uuid2}.sif")
	cmd_no_output(f"singularity build {sif_path} docker-archive://{uuid2}.tar")
	cmd_no_output(f"rm {uuid2}.tar")
	
	return sif_path
		
def param_str(k, v, widget_conf):
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


def get_job_cmd(widget_dir, props, inputs):
	widget_name = os.path.basename(widget_dir)
	widget_json_path = os.path.join(widget_dir, f"{widget_name}.json")
	with open(widget_json_path, 'r') as f:
		widget_json_raw = f.read()
	widget_conf = jsonpickle.decode(widget_json_raw)

	command = widget_conf['command'][0] + ' '

	# Way to prevent downloadUrl widget from redownloading massive
	# fastq files. Remove this later.
	if widget_name == "downloadURL":
		command += ' --noClobber '

	required_params = widget_conf['requiredParameters']

	for k in required_params:
		if k not in inputs:
			command += param_str(k, props[k], widget_conf)

	for k in inputs:
		command += param_str(k, inputs[k], widget_conf)
	
	params = widget_conf['parameters']
	for k in params:
		if k in required_params or k in inputs:
			continue
		if k in props:
			command += param_str(k, props[k], widget_conf)
		elif 'default' in params[k] and  params[k]['default']:
			command += param_str(k, params[k]['default'], widget_conf)


	return command


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

def update_job_inputs(job_id, outlinks, src_outputs, inputs_dict):
	for dst_node, outputs in outlinks.items():
		for output in outputs:
			src_name, dst_name = output

			# Noticing some odd behavior here. Inputs which are
			# specified via links in "OWS" file do not exist as
			# output files in /tmp/output of source node contai
			# ner.
			if src_name in src_outputs:
				inputs_dict[dst_node][dst_name] = src_outputs[src_name]
	

def gen_workflow_command(workflow_dir):
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
	
	# I've selected an ordering of jobs for salmon_demo
	# to help speed up testing. Be sure to remove this later.
	# job_top_sort = ['3', '5', '4', '2']

	inputs = {}
	for job_id in job_top_sort:
		inputs[job_id] = {}

	for job_id in job_top_sort:
		print(f"Executing job ID {job_id}")
		job_name = job_id_to_name[job_id]
		job_dir = os.path.join(widget_dir, job_name)

		job_img_path = build_singularity_img(job_dir)
		job_cmd = get_job_cmd(job_dir, job_props[job_id], inputs[job_id])

		run_sif_cmd(job_img_path, job_cmd)
		outputs = get_sif_outputs(job_img_path)
		update_job_inputs(job_id, job_io_links[job_id], outputs, inputs)

	widgets_dir = os.path.join(workflow_dir, "widgets")
	widget_child_dir = [ f for f in os.scandir(widgets_dir) if f.is_dir() ]
	widget_dir = widget_child_dir[0]

	

if __name__ == "__main__":
	gen_workflow_command(sys.argv[1])
