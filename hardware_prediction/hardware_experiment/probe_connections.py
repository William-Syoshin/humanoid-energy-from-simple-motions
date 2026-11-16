"""Read-only robot probe and short ESP acquisition; no servo commands."""
import sys,time,argparse
from pathlib import Path
import serial
from core import read,save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);a=parser.parse_args()
    cfg=read(Path(__file__).parent/'session_template/config.json')
    report={'servo_commands_sent':False,'esp_lines':[],'joints':[]}
    with serial.Serial(cfg['power_port'],cfg['power_baud'],timeout=.2) as port:
        try:
            time.sleep(2);port.write(b'START\n');end=time.perf_counter()+2
            while time.perf_counter()<end:
                line=port.readline().decode(errors='replace').strip()
                if line:report['esp_lines'].append({'pc_s':time.perf_counter(),'line':line})
        finally:port.write(b'STOP\n')
    sys.path.insert(0,cfg['rcb_library'])
    from Rcb4BaseLib import Rcb4BaseLib
    rcb=Rcb4BaseLib()
    try:
        report['robot_open']=bool(rcb.open(cfg['robot_port'],cfg['robot_baud'],.15))
        if report['robot_open']:
            for j in cfg['joints']:
                ok,pos=rcb.getSinglePos(j['id'],j['sio'])
                report['joints'].append({'name':j['name'],'id':j['id'],'sio':j['sio'],'ok':bool(ok),'raw_position':pos})
            ok,data=rcb.moveRamToComCmdSynchronize(rcb.adDataAddr(1),8)
            report['ad_ok']=bool(ok);report['ad_bytes']=list(data)
    finally:rcb.close()
    save(a.out,report)
    print(report)

if __name__=='__main__':main()
