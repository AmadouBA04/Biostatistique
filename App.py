import tornado.websocket
import tornado.web
import tornado.ioloop

class MyWebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, *args, **kwargs):
        # Initialisation d'autres variables ou processus ici
        super().initialize(*args, **kwargs)

    def open(self):
        # Code exécuté lorsqu'une connexion WebSocket est ouverte
        self.write_message("Bienvenue sur WebSocket!")

    def on_message(self, message):
        # Code pour gérer les messages reçus
        self.write_message(f"Message reçu : {message}")

    def on_close(self):
        # Code exécuté lors de la fermeture de la connexion
        print("Connexion fermée.")

def make_app():
    return tornado.web.Application([
        (r"/websocket", MyWebSocketHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
