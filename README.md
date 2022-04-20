
## Tool setup

* Create a virtual env specific for the tool
* Run `pip install -r <tool>/requirements.txt`

## Tool execution

```shell
python <toolname>/main.py github|jira
```

### Ludwig

```shell
ludwig experiment --dataset=data/jira|github.csv --config_file=ludwig/config.yaml 
```
