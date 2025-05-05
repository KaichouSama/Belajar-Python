/*!
* Start Bootstrap - Simple Sidebar v6.0.6 (https://startbootstrap.com/template/simple-sidebar)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-simple-sidebar/blob/master/LICENSE)
*/
// 
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        // Uncomment Below to persist sidebar toggle between refreshes
        // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
        //     document.body.classList.toggle('sb-sidenav-toggled');
        // }
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }

});

function appendMessage(text, className) {
    const chatbox = document.getElementById("chatbox");
    const div = document.createElement("div");
    div.className = className + " message";
    div.innerText = text;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
  }

  function kirimChat() {
    const input = document.getElementById("chatInput");
    const msg = input.value.trim();
    if (!msg) return;

    appendMessage(msg, "user-message");
    input.value = "";

    fetch('/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    })
      .then(res => res.json())
      .then(data => {
        appendMessage(data.response, "bot-message");
      })
      .catch(() => {
        document.getElementById("typingIndicator").style.display = "none";
        appendMessage("Maaf, terjadi kesalahan.", "bot-message");
      });
  }