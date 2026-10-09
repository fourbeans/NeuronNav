#!/usr/bin/env python3
"""Compare two PONI evaluation Namespace records, excluding intentional paths."""

import argparse
import json
import re


def namespace(path):
    with open(path) as handle:
        text = handle.read()
    match = re.search(r"Namespace\((.*?)\)\n", text, re.DOTALL)
    if not match:
        raise RuntimeError("No Namespace record in {}".format(path))
    fields = {}
    for item in match.group(1).split(", "):
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = namespace(args.reference)
    candidate = namespace(args.candidate)
    ignored = ("dump_location", "pf_model_path")
    ref_protocol = {key: value for key, value in reference.items() if key not in ignored}
    new_protocol = {key: value for key, value in candidate.items() if key not in ignored}
    differing = {
        key: [ref_protocol.get(key), new_protocol.get(key)]
        for key in sorted(set(ref_protocol) | set(new_protocol))
        if ref_protocol.get(key) != new_protocol.get(key)
    }
    report = {
        "protocol_match": not differing,
        "ignored_intentional_fields": {
            key: [reference.get(key), candidate.get(key)] for key in ignored
        },
        "differing_protocol_fields": differing,
        "compared_field_count": len(set(ref_protocol) | set(new_protocol)),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    with open(args.output, "w") as handle:
        handle.write(rendered + "\n")
    if differing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
