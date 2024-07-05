from image_processor import ImageProcessor
from memory_manager import MemoryManager
from artmuse import ArtMuse
from artmuse import ConversationManager    

def main():
    memory_manager = MemoryManager()
    image_processor = ImageProcessor()
    art_muse = ArtMuse(memory_manager, image_processor)
    conversation_manager = ConversationManager(art_muse)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = conversation_manager.manage_conversation(user_input)
        print("ArtMuse:", response)

if __name__ == "__main__":
    main()