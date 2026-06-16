from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


MODEL_PATH = Path(__file__).resolve().parent.parent / 'data' / 'gait2d.xml'


def _parse_vector(value, expected_length=None):
    if value is None:
        if expected_length is None:
            return np.array([], dtype=float)
        return np.zeros(expected_length, dtype=float)
    vector = np.fromstring(value, sep=' ', dtype=float)
    if expected_length is not None and vector.size == 0:
        return np.zeros(expected_length, dtype=float)
    return vector


def get_model_dictionary():
    tree = ET.parse(MODEL_PATH)

    def parse_body(body, parent=None):
        name = body.attrib['name']
        body_info = {
            'parent': parent,
            'joint_location': _parse_vector(body.attrib.get('pos'), expected_length=3),
            'orientation': _parse_vector(body.attrib.get('quat'), expected_length=4),
            'joints': {},
            'markers': {},
            'children': {},
        }

        for joint in body.findall('joint'):
            joint_name = joint.attrib['name']
            joint_type = joint.attrib.get('type', '')
            if joint_type == 'slide':
                joint_type = 'slider'

            limits = None
            if 'range' in joint.attrib:
                limits = _parse_vector(joint.attrib['range'], expected_length=2)

            body_info['joints'][joint_name] = {
                'type': joint_type,
                'axis': _parse_vector(joint.attrib.get('axis'), expected_length=3),
                'limits': limits,
                'pos': _parse_vector(joint.attrib.get('pos'), expected_length=3),
            }

        for site in body.findall('site'):
            site_name = site.attrib.get('name')
            if not site_name or not site_name.startswith('marker_'):
                continue
            body_info['markers'][site_name] = _parse_vector(site.attrib.get('pos'), expected_length=3)

        inertial = body.find('inertial')
        if inertial is not None:
            body_info['mass'] = float(inertial.attrib.get('mass', 0.0))
            body_info['com'] = _parse_vector(inertial.attrib.get('pos'), expected_length=3)
            body_info['inertia'] = _parse_vector(inertial.attrib.get('fullinertia'), expected_length=6)
        else:
            body_info['mass'] = 0.0
            body_info['com'] = np.zeros(3, dtype=float)
            body_info['inertia'] = np.zeros(6, dtype=float)

        for child in body.findall('body'):
            child_info = parse_body(child, name)
            body_info['children'][child.attrib['name']] = child_info

        return body_info

    root_body = tree.getroot().find('worldbody/body')
    if root_body is None:
        return {}

    return {root_body.attrib['name']: parse_body(root_body)}


kintree = get_model_dictionary()