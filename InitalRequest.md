
# Eujeno

Progetta un'architettura per un sistema di inferenza distribuito peer-to-peer per modelli di linguaggio di grandi dimensioni. Il sistema deve essere completamente decentralizzato senza server centrale.

1. Ogni nodo è un peer che può entrare e uscire dalla rete dinamicamente.

2. Ogni nodo si registra per processare uno o più layer specifici di un modello LLM suddiviso. 

3. Quando un utente fa una domanda, il sistema instrada i vettori attraverso la rete dai layer responsabili.

Implementa un meccanismo di discovery distribuito — potrebbe essere un DHT o blockchain-based — che traccia quale nodo gestisce quale layer, lo stato dei nodi, e la reputazione. 
Gestisci la ridondanza: più nodi possono processare lo stesso layer per resilienza. Se un nodo cade, il traffico si reindirizza automaticamente.

Implementa un sistema di incentivi con token o criptovalute che ricompensa i nodi per il tempo di calcolo e larghezza di banda forniti. 
Supporta latenze alte, le risposte potrebbero arrivare dopo ore, giorni, o settimane. Non è real-time.

Fornisci load balancing intelligente: richieste diverse si accodano ai layer specifici, massimizzando l'utilizzo della rete.

Il sistema deve integrarsi con modelli open source scaricabili da hugging faces. 

Quando un nodo si unisce, scarica il modello completo, lo splitta nei layer, e registra quali layer può processare. L'allocazione dei layer è dinamica — nuovi nodi ricevono i layer non ancora coperti in base a logica di bilanciamento. 

Man mano che i nodi si aggiungono, il modello si compone progressivamente nella rete.

Il modello diventa operativo soltanto quando tutti i layer sono allocati ad almeno un nodo. Prima di questo, le richieste vengono messe in coda. Una volta completato il deployment, il sistema inizia a processare le domande degli utenti attraverso l'intera rete di layer distribuiti.

Considera sicurezza, Byzantine fault tolerance, e gestione dei fallimenti parziali della rete.

Esplora l'architettura completa, disegna i componenti principali, suggerisci stack tecnologico, e sviluppa codice per: un nodo peer che si registra e processa layer, un meccanismo di discovery e routing, un sistema di queue e load balancing, un layer di incentivizzazione e reputazione.

Estendi questa PRD magari splittandola in varie parti qalora servisse affrontare questo problema complesso in maniera comparimentalizzata. 
Lancia un team di agent per confrontare varie strade di implementazione in modo poi da scegliere quella migliore.

Puoi mantenere questo progetto su un repository di github per adesso privato poi ovviamente non appena abbiamo qualcosa che funziona, dovra essere pubblico per richiamare piu gente possibile all'esperimento.