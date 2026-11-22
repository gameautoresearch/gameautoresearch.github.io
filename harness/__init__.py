"""The research-agent harness for all six games.

Harbor's Terminus-2 is the researcher. It works in one uniform workspace and
talks to the host through the `ar` command. Two task variants share that setup:
`recipe` (the final source/config is retrained from scratch and graded) and
`network` (a network trained during the session is graded directly).
"""
