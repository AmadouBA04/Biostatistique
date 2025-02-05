import tornado.ioloop
import tornado.web
import tornado.websocket

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        self.write_message("Bienvenue dans le WebSocket!")

    def on_message(self, message):
        self.write_message(f"Message reçu : {message}")

    def on_close(self):
        print("Connexion fermée")

def make_app():
    return tornado.web.Application([
        (r"/websocket", WebSocketHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)  # Serveur écoute sur le port 8888
    tornado.ioloop.IOLoop.current().start()
