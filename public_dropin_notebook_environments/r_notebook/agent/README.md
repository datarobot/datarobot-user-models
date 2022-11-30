# Kernel Agent

A light-weight agent service running on the Kernel container

## Deployment

The agent is installed during image build as defined in the Dockerfile.

### General purpose agent

Other endpoints may be added for features that require direct access to the Kernel container.
Currently, the agent is only used to push resource utilization metrics to the runner. 
The only endpoint exposed is a WebSocket route that the runner can connect to. 
