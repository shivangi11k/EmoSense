const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');
const container = document.getElementById('container');

signUpButton.addEventListener('click', () => {
    container.classList.add("right-panel-active");
});

signInButton.addEventListener('click', () => {
    container.classList.remove("right-panel-active");
});

// Redirect user after successful sign up
const signUpForm = document.querySelector('.sign-up-container form');
signUpForm.addEventListener('submit', (e) => {
    e.preventDefault(); // Prevent form submission

    // Perform sign up logic here

    // Redirect to the next page after successful sign up
    window.location.href = 'next-page.html'; // Replace 'next-page.html' with your desired page URL
});

