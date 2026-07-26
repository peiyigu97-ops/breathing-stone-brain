"""
bore-protocol reverse tunnel client.
Connects to bore.pub:7835, registers a port, then relays TCP traffic
bidirectionally between bore.pub and localhost:7860.

Protocol (bore v0.5):
  Client → Server: Hello { local_port: u16, secret: Option<String> }
  Server → Client: Ack { remote_port: u16 }  (on success)
                   Error { reason: String }   (on failure)
  Server → Client: Connection { id: Uuid }    (each new inbound conn)
  Client → Server: Accept { id: Uuid }
  Then raw bidirectional relay on the same TCP stream.
"""
import socket, threading, struct, json, uuid, sys, time, os

BORE_HOST = "bore.pub"
BORE_PORT = 7835
LOCAL_PORT = 7860

def send_msg(sock, obj):
    data = json.dumps(obj).encode()
    # 4-byte big-endian length prefix
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock):
    raw = b""
    while len(raw) < 4:
        chunk = sock.recv(4 - len(raw))
        if not chunk:
            return None
        raw += chunk
    length = struct.unpack(">I", raw)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode())

def relay(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try: src.close()
        except: pass
        try: dst.close()
        except: pass

def handle_connection(conn_id):
    try:
        # Open a new TCP connection to bore.pub for this relay
        relay_sock = socket.create_connection((BORE_HOST, BORE_PORT), timeout=10)
        send_msg(relay_sock, {"type": "Accept", "id": conn_id})

        # Connect to local server
        local_sock = socket.create_connection(("127.0.0.1", LOCAL_PORT), timeout=5)

        # Bidirectional relay
        t1 = threading.Thread(target=relay, args=(relay_sock, local_sock), daemon=True)
        t2 = threading.Thread(target=relay, args=(local_sock, relay_sock), daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
    except Exception as e:
        print(f"  relay error: {e}")

def run():
    while True:
        try:
            print(f"Connecting to {BORE_HOST}:{BORE_PORT}...")
            ctrl = socket.create_connection((BORE_HOST, BORE_PORT), timeout=10)
            send_msg(ctrl, {"type": "Hello", "local_port": LOCAL_PORT})

            msg = recv_msg(ctrl)
            if msg is None:
                print("No response, retrying..."); time.sleep(3); continue

            if msg.get("type") == "Error":
                print("Server error:", msg.get("reason")); time.sleep(3); continue

            if msg.get("type") == "Ack":
                remote_port = msg["remote_port"]
                url = f"http://{BORE_HOST}:{remote_port}"
                print(f"\n{'='*60}")
                print(f"  PUBLIC URL: {url}")
                print(f"{'='*60}\n")

                # Listen for inbound connection requests
                while True:
                    conn_msg = recv_msg(ctrl)
                    if conn_msg is None:
                        print("Control connection lost, reconnecting...")
                        break
                    if conn_msg.get("type") == "Connection":
                        cid = conn_msg["id"]
                        print(f"  New connection: {cid[:8]}...")
                        t = threading.Thread(target=handle_connection, args=(cid,), daemon=True)
                        t.start()
            else:
                print("Unexpected response:", msg); time.sleep(3)
        except Exception as e:
            print(f"Connection error: {e}, retrying in 3s...")
            time.sleep(3)

if __name__ == "__main__":
    run()
