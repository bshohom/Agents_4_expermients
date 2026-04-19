# Experiment Schema Notes

This file is intentionally a placeholder.

A future coding step should define a structured schema for proposed experiments,
for example with Pydantic, including fields like:

- proposed_experiment
- source_titles
- required_equipment
- controllable_parameters
- expected_measurements
- bom_fit_reasoning
- unsupported_requirements
- confidence_notes

This schema should then be used to validate LLM outputs before downstream use.
