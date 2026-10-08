"""UI protocol fixture only. No model, no meaningful performance numbers.

Run from project root: PYTHONPATH=. python3 tests/manual_service.py
Uses isolated temporary state, app :8766 and fixture :18081.
"""
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from workbench.server import App, make_server


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
        if self.path=='/metrics':
            self.wfile.write(b'vllm:kv_cache_usage_perc{model_name="SYNTHETIC-FIXTURE"} 0.25\n');return
        self.wfile.write(b'{"data":[{"id":"protocol-fixture-not-a-model"}]}')
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert body['model']=='protocol-fixture-not-a-model'
        self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
        try:
            for word in ['Protocol fixture only. ', 'This is not a model response. ', 'Received %s messages.' % len(body['messages'])]:
                obj={'choices':[{'delta':{'content':word}}]}
                self.wfile.write(b'data: '+json.dumps(obj).encode()+b'\n\n'); self.wfile.flush(); time.sleep(.6)
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError): pass


if __name__=='__main__':
    fixture=ThreadingHTTPServer(('127.0.0.1',18081),Fixture)
    threading.Thread(target=fixture.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='workbench-ui-') as state:
        app=App(state,[state]); server=make_server(app,8766)
        app.services.register({'url':'http://127.0.0.1:18081','model':'protocol-fixture-not-a-model','name':'SYNTHETIC protocol fixture'})
        from pathlib import Path
        trace=Path(state)/'SYNTHETIC-ui-test.json'
        trace.write_text(json.dumps({'traceEvents':[{'ph':'X','name':'SYNTHETIC aten::mm','cat':'cpu_op','pid':1,'tid':1,'ts':100,'dur':50,'args':{'Input Dims':[[2,3],[3,4]]}},{'ph':'X','name':'SYNTHETIC kernel','cat':'kernel','pid':2,'tid':7,'ts':110,'dur':30}]}))
        print('TEST TRACE '+str(trace),flush=True)
        print('ISOLATED PROTOCOL TEST http://127.0.0.1:8766/#token='+app.token,flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally:
            app.services.shutdown(); app.downloads.shutdown(); app.jobs.shutdown()
            server.server_close(); fixture.shutdown(); fixture.server_close()
