use gtk::prelude::*;
use gtk::{Application, ApplicationWindow, Box, Button, Orientation};
use std::collections::HashMap;
use std::thread;

mod ai {
    use tch::{nn, Tensor};

    pub struct Model {
        model: nn::Module,
    }

    impl Model {
        pub fn new() -> Self {
            let model = nn::seq()
                .add(nn::linear(1024, 256, Default::default()))
                .add(nn::relu())
                .add(nn::linear(256, 128, Default::default()))
                .add(nn::relu())
                .add(nn::linear(128, 2, Default::default()));
            Self { model }
        }

        pub fn predict(&self, input: &Tensor) -> Tensor {
            self.model.forward(input)
        }
    }
}

fn main() {
    gtk::prelude::init();

    let application =
        Application::new(Some("com.81au.security_tool"), Default::default())
            .expect("failed to create application");

    application.connect_activate(build_ui);

    application.run(Default::default());

    thread::spawn(move || {
        let model = ai::Model::new();
        let mut inputs = HashMap::new();

        loop {
            // simulate data input
            inputs.insert("sensor1".to_string(), 10.0);
            inputs.insert("sensor2".to_string(), 20.0);

            let input = Tensor::of_slice(&[inputs["sensor1"](), inputs["sensor2"]()]);
            let output = model.predict(&input);

            // simulate anomaly detection
            if output[0][0] > 0.5 {
                println!("ANOMALY DETECTED!");
            }

            thread::sleep(std::time::Duration::from_millis(1000));
        }
    });
}

fn build_ui(application: &gtk::Application) {
    let window = ApplicationWindow::new(application, Default::default());
    window.set_title("AI-Powered Security Tool");
    window.set_default_size(800, 600);

    let vbox = Box::new(Orientation::Vertical, 0);
    window.add(&vbox);

    let button = Button::with_label("Start Monitoring");
    vbox.add(&button);

    button.connect_clicked(move |_| {
        println!("Monitoring started...");
    });

    window.show_all();
}