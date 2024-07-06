from typing import Optional
from paramiko import SSHClient
from scp import SCPClient


def send(local_path:str, remote_path:str, 
         remote_ip:str, port:int=22, username:Optional[str]=None, password:Optional[str]=None, key_filename:Optional[str]=None) -> None:
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(remote_ip, port=port, username=username, password=password, key_filename=key_filename)

    scp = SCPClient(ssh.get_transport())

    scp.put(local_path, remote_path)

    scp.close()

def read(local_path:str, remote_path:str, 
         remote_ip:str, port:int=22, username:Optional[str]=None, password:Optional[str]=None, key_filename:Optional[str]=None) -> None:
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(remote_ip, port=port, username=username, password=password, key_filename=key_filename)

    scp = SCPClient(ssh.get_transport())

    scp.get(remote_path, local_path)

    scp.close()
