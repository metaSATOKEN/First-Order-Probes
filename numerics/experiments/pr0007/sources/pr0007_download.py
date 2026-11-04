# Copyright 2017 Google Inc. (Colab comm-based transport pattern)
# Copyright 2026 Kentaro Sato (serial queue, acknowledgements, cleanup and streaming upload)
# SPDX-License-Identifier: Apache-2.0
# Adapted from googlecolab/colabtools/google/colab/files.py; see PR0007_THIRD_PARTY_NOTICE.txt.
"""One browser transfer at a time; browser receipt is NOT local-disk confirmation."""
import base64
import html
import json
from pathlib import Path
import pkgutil
import uuid
from pr0007_io import write,read,stamp,digest,ZIP_LIMIT

QUEUE_JS = r'''
globalThis.pr0007Queue = globalThis.pr0007Queue || {
  items: [], keys: new Set(), running: false, paused: false, maxActive: 0, active: 0,
  async drain() {
    if (this.running || this.paused) return;
    this.running = true;
    try {
      while (this.items.length && !this.paused) {
        const item = this.items[0];
        let buffers = [], url = null, channel = null, received = 0, saved = false;
        const label = document.createElement('div');
        label.textContent = 'PR-0007 download: ' + item.name;
        document.body.appendChild(label);
        this.active++; this.maxActive = Math.max(this.maxActive, this.active);
        const timed = (promise, ms) => {
          let timer;
          return Promise.race([promise, new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error('Browser transfer timeout')), ms);
          })]).finally(() => clearTimeout(timer));
        };
        try {
          if (!google.colab.kernel.accessAllowed) throw new Error('Browser kernel access denied');
          channel = await timed(google.colab.kernel.comms.open(item.id), 90000);
          const it = channel.messages[Symbol.asyncIterator]();
          channel.send({action:'next'});
          while (received < item.size) {
            const next = await timed(it.next(), 90000);
            if (next.done) throw new Error('Truncated browser transfer');
            for (const buffer of (next.value.buffers || [])) {
              buffers.push(buffer); received += buffer.byteLength;
            }
            if (received > item.size) throw new Error('Oversized browser transfer');
            label.textContent = item.name + ': ' + (100*received/item.size).toFixed(1) + '% received';
            if (received < item.size) channel.send({action:'next'});
          }
          const blob = new Blob(buffers, {type:'application/zip'});
          url = URL.createObjectURL(blob);
          const link = document.createElement('a'); link.href=url; link.download=item.name;
          label.appendChild(link); link.click(); saved=true;
          // Release references before the next ZIP. A download click is not a disk-save receipt.
          await new Promise(resolve => setTimeout(resolve, 10000));
          URL.revokeObjectURL(url); url=null; buffers.length=0;
          channel.send({action:'complete',received_bytes:received,save_requested:true,resources_released:true});
          this.items.shift(); this.keys.delete(item.key);
          label.textContent = item.name + ': browser received / save requested (disk confirmation unavailable)';
        } catch (error) {
          this.paused=true;
          if (channel) channel.send({action:'cancel',message:String(error),received_bytes:received,save_requested:saved});
          label.textContent = 'Download queue PAUSED: '+String(error)+'. Runtime files retained; rerun cell to retry.';
          this.items.shift(); this.keys.delete(item.key);
        } finally {
          if (url) URL.revokeObjectURL(url);
          buffers.length=0; channel=null; this.active--;
        }
      }
    } finally { this.running=false; }
  },
  enqueue(item) {
    if (this.keys.has(item.key)) return;
    this.keys.add(item.key); this.items.push(item); this.drain();
  }
};
'''


class Delivery:
    def __init__(self,folder,enabled=True):
        self.folder=Path(folder);self.enabled=enabled;self.enqueued=set()
        self.folder.mkdir(parents=True,exist_ok=True)
        if enabled:
            from IPython.display import display,Javascript
            display(Javascript(QUEUE_JS+'\npr0007Queue.paused=false; pr0007Queue.drain();'))

    def enqueue(self,path):
        path=Path(path)
        if not path.exists() or path.stat().st_size>=ZIP_LIMIT:raise ValueError('Invalid bounded ZIP')
        key=digest(path);receipt=self.folder/(key+'.json')
        prior=read(receipt) if receipt.exists() else {}
        if prior.get('browser_received') and prior.get('save_requested'):return
        if key in self.enqueued:return
        self.enqueued.add(key)
        record=dict(path=str(path),name=path.name,sha256=key,bytes=path.stat().st_size,
            computed=True,zip_verified=True,browser_received=False,save_requested=False,
            user_disk_confirmed=False,at=stamp(),state='queued' if self.enabled else 'disabled')
        write(receipt,record)
        if not self.enabled:return
        import IPython
        from IPython.display import display,Javascript
        manager=IPython.get_ipython().kernel.comm_manager;comm_id='pr0007_'+uuid.uuid4().hex
        def target(comm,_):
            f=path.open('rb');sent=0
            def close():
                f.close();comm.close();manager.unregister_target(comm_id,target)
            def message(msg):
                nonlocal sent
                data=msg['content']['data'];action=data.get('action')
                if action=='next':
                    chunk=f.read(1024**2);sent+=len(chunk)
                    if chunk:comm.send({},None,[chunk])
                    else:close()
                elif action in ('complete','cancel'):
                    ok=action=='complete' and sent==record['bytes']==data.get('received_bytes') and data.get('resources_released') is True
                    record.update(browser_received=bool(ok),save_requested=bool(ok and data.get('save_requested')),
                        state='save_requested' if ok else 'paused',at=stamp(),reported=data)
                    write(receipt,record);close()
            comm.on_msg(message)
        manager.register_target(comm_id,target)
        item=dict(id=comm_id,key=key,name=path.name,size=path.stat().st_size)
        display(Javascript('pr0007Queue.enqueue('+json.dumps(item)+');'))

    def status(self):
        return dict(at=stamp(),items=[read(p) for p in sorted(self.folder.glob('*.json'))],
            limitation='Browser acknowledgement is not proof of saved bytes on the user disk. Pending states are snapshots.')


def upload_streaming(destination):
    """Colab's upload JS generator, but write each fragment rather than accumulating all ZIPs."""
    from google.colab import output
    from IPython.display import display,HTML
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex;input_id='files-'+token;output_id='result-'+token
    js=pkgutil.get_data('google.colab','resources/files.js').decode('utf8')
    display(HTML(f'<input type="file" id="{input_id}" multiple disabled style="border:none" />'
        f'<output id="{output_id}">Select the relevant PR-0007 ZIPs.</output><script>{js}</script>'))
    result=output.eval_js(f'google.colab._files._uploadFiles("{input_id}","{output_id}")')
    paths={};sizes={}
    while result['action']!='complete':
        result=output.eval_js(f'google.colab._files._uploadFilesContinue("{output_id}")')
        if result['action']!='append':continue
        name=result['file']
        if Path(name).name!=name or not name.endswith('.zip'):raise ValueError('Choose ZIP files only')
        if name not in paths:
            p=destination/(token+'_'+name);p.touch(exist_ok=False);paths[name]=p;sizes[name]=0
        fragment=base64.b64decode(result['data'],validate=True);sizes[name]+=len(fragment)
        if sizes[name]>=ZIP_LIMIT:raise ValueError('Uploaded ZIP >= decimal 1GB')
        with paths[name].open('ab') as f:f.write(fragment)
    return list(paths.values())
