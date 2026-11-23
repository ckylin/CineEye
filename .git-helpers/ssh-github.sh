#!/bin/bash
# 影瞳项目 GitHub SSH 包装器：绕过 22 端口封锁与 DNS 劫持
# 走 GitHub 官方 ssh.github.com:443 + 本机 HTTP 代理 CONNECT 隧道
#
# git 调用格式: ssh <user@host> <git 子命令...>
# 本包装器固定连 ssh.github.com:443（git 用户），只转发子命令部分。
# -F /dev/null 绕开本机有权限问题的 /etc/ssh/ssh_config.d/

# 去掉第一个参数（git@github.com 或 ssh.github.com），其余是真正的 git 子命令
shift

exec ssh -F /dev/null \
  -o ProxyCommand='ncat --proxy 127.0.0.1:37851 --proxy-type http %h %p' \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/dev/null \
  -o User=git -i /home/kylin/.ssh/id_rsa \
  -p 443 ssh.github.com "$@"
