const displayLoader = (shouldDisplay) => {
  // remove existing loaders
  const elements = document.querySelectorAll(`.loader`);
  if (shouldDisplay && elements.length == 0) {
    const c = document.querySelector(".arc_reactor_container");
    loader = document.createElement("span");
    loader.classList.add("loader");
    c.appendChild(loader);
  } else {
    elements.forEach((element) => {
      element.remove();
    });
  }
};

const displayLine = (isUser, message) => {
  const chatList = document.querySelector(".chat_list");

  // Create the list item
  const listItem = document.createElement("li");
  listItem.classList.add("chat_list_item");

  // Create the keyword (USER or JARVIS)
  const keyword = document.createElement("p");
  keyword.classList.add(isUser ? "keyword-user" : "keyword-jarvis");
  keyword.textContent = isUser ? "USER: " : "JARVIS: ";

  // Create the message content
  const normalWord = document.createElement("p");
  normalWord.classList.add("normal_word");
  normalWord.textContent = message;

  // Append keyword and message to the list item
  listItem.appendChild(keyword);
  listItem.appendChild(normalWord);

  // Append the list item to the chat list
  chatList.appendChild(listItem);

  // Check if there are more than 6 items in the list
  if (chatList.children.length > 6) {
    // Remove the first item (FIFO - First In, First Out)
    chatList.removeChild(chatList.children[0]);
  }
};
